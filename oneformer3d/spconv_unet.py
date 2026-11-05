import functools   # c2
from collections import OrderedDict

import spconv.pytorch as spconv
import torch
from spconv.pytorch.modules import SparseModule
from torch import nn

from mmdet3d.registry import MODELS


class ResidualBlock(SparseModule):
    """Residual block for SpConv U-Net."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_fn=functools.partial(
                     nn.BatchNorm1d, eps=1e-4, momentum=0.1),
                 indice_key=None,
                 normalize_before=True):
        super().__init__()

        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(nn.Identity())
        else:
            self.i_branch = spconv.SparseSequential(
                spconv.SubMConv3d(
                    in_channels, out_channels, kernel_size=1, bias=False))

        if normalize_before:
            self.conv_branch = spconv.SparseSequential(
                norm_fn(in_channels), nn.ReLU(),
                spconv.SubMConv3d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    indice_key=indice_key),
                norm_fn(out_channels), nn.ReLU(),
                spconv.SubMConv3d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    indice_key=indice_key))
        else:
            self.conv_branch = spconv.SparseSequential(
                spconv.SubMConv3d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    indice_key=indice_key),
                norm_fn(out_channels), nn.ReLU(),
                spconv.SubMConv3d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    indice_key=indice_key),
                norm_fn(out_channels), nn.ReLU())

    def forward(self, input):
        identity = spconv.SparseConvTensor(
            input.features,
            input.indices,
            input.spatial_shape,
            input.batch_size)

        output = self.conv_branch(input)
        output = output.replace_feature(
            output.features + self.i_branch(identity).features)
        return output


class GAASEBlock(SparseModule):
    """Geometry-Guided Adaptive Anisotropic Sparse Residual Block.

    Design goal:
        - Replace ONLY the first ResidualBlock in encoder stages E1/E2.
        - Keep the second 3x3x3 sparse convolution and the residual path.
        - Use a geometry-guided three-branch first convolution:
              isotropic 3x3x3
              Z-direction 1x1x3
              XY-plane    3x3x1
        - Geometry is estimated directly from sparse occupancy at the current
          encoder resolution, so no extra dataset fields / KNN preprocessing
          are required.

    IMPORTANT coordinate convention:
        The current collate code constructs coordinates as [batch, x, y, z].
        Therefore the 3rd spatial kernel dimension is treated as physical Z:
            Z branch  = (1, 1, 3)
            XY branch = (3, 3, 1)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_fn=functools.partial(
                     nn.BatchNorm1d, eps=1e-4, momentum=0.1),
                 indice_key=None,
                 normalize_before=True,
                 gate_hidden=16,
                 iso_init_bias=1.5):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize_before = normalize_before
        self.indice_key = indice_key or 'gaase'

        # Residual / identity path.
        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(nn.Identity())
        else:
            self.i_branch = spconv.SparseSequential(
                spconv.SubMConv3d(
                    in_channels, out_channels, kernel_size=1, bias=False))

        # Match the original ResidualBlock ordering as closely as possible.
        self.relu = nn.ReLU()
        if normalize_before:
            self.pre_norm = norm_fn(in_channels)
            self.mid_norm = norm_fn(out_channels)
        else:
            self.post_norm1 = norm_fn(out_channels)
            self.post_norm2 = norm_fn(out_channels)

        # --------------------------------------------------------------
        # First convolution -> geometry-guided anisotropic three branches.
        # --------------------------------------------------------------
        iso_key = f'{self.indice_key}_gaase_iso'
        z_key = f'{self.indice_key}_gaase_z'
        xy_key = f'{self.indice_key}_gaase_xy'

        self.conv_iso = spconv.SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            indice_key=iso_key)

        self.conv_z = spconv.SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=(1, 1, 3),
            padding=(0, 0, 1),
            bias=False,
            indice_key=z_key)

        self.conv_xy = spconv.SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=(3, 3, 1),
            padding=(1, 1, 0),
            bias=False,
            indice_key=xy_key)

        # --------------------------------------------------------------
        # Fixed occupancy convolutions for LOCAL GEOMETRY.
        # They use all-one kernels and are frozen.
        # Geometry descriptor per active voxel:
        #   g0 = 3D occupancy density in 3x3x3
        #   g1 = Z-line occupancy density in 1x1x3
        #   g2 = XY-plane occupancy density in 3x3x1
        #   g3 = directional contrast = g1 - g2
        # --------------------------------------------------------------
        self.geo_iso = spconv.SubMConv3d(
            1, 1, kernel_size=3, padding=1, bias=False,
            indice_key=iso_key)
        self.geo_z = spconv.SubMConv3d(
            1, 1, kernel_size=(1, 1, 3), padding=(0, 0, 1),
            bias=False, indice_key=z_key)
        self.geo_xy = spconv.SubMConv3d(
            1, 1, kernel_size=(3, 3, 1), padding=(1, 1, 0),
            bias=False, indice_key=xy_key)
        self._freeze_geometry_kernels()

        # Geometry + current learned feature jointly generate branch weights.
        self.geometry_embed = nn.Sequential(
            nn.Linear(4, gate_hidden),
            nn.GELU())
        self.feature_embed = nn.Sequential(
            nn.Linear(in_channels, gate_hidden),
            nn.GELU())
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_hidden * 2, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 3))

        # Stable initialization: start biased toward the original 3D branch,
        # then let geometry/feature evidence learn to increase Z / XY branches.
        with torch.no_grad():
            nn.init.zeros_(self.gate_mlp[-1].weight)
            self.gate_mlp[-1].bias.copy_(
                torch.tensor([iso_init_bias, 0.0, 0.0], dtype=torch.float32))

        # Keep the SECOND convolution of the original ResidualBlock unchanged.
        self.conv2 = spconv.SubMConv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            indice_key=self.indice_key)

        # Debug/analysis values; not used in the forward computation.
        self.last_gate_mean = None
        self.last_geometry_mean = None

    def _freeze_geometry_kernels(self):
        """Make occupancy kernels fixed all-one filters."""
        for conv in (self.geo_iso, self.geo_z, self.geo_xy):
            with torch.no_grad():
                conv.weight.fill_(1.0)
            conv.weight.requires_grad_(False)

    def _compute_geometry(self, x):
        """Compute local sparse-occupancy geometry at current resolution."""
        # Defensive reset: keeps the geometry filters fixed even if an external
        # initializer touches registered module weights.
        self._freeze_geometry_kernels()

        ones = x.features.new_ones((x.features.shape[0], 1))
        occupancy = x.replace_feature(ones)

        n_iso = self.geo_iso(occupancy).features
        n_z = self.geo_z(occupancy).features
        n_xy = self.geo_xy(occupancy).features

        # Normalize by the maximum number of positions in each kernel.
        rho_iso = n_iso / 27.0
        rho_z = n_z / 3.0
        rho_xy = n_xy / 9.0
        contrast = rho_z - rho_xy

        geometry = torch.cat(
            [rho_iso, rho_z, rho_xy, contrast], dim=1)
        return geometry

    def _anisotropic_first_conv(self, x):
        """Geometry-guided adaptive three-branch sparse convolution."""
        geometry = self._compute_geometry(x)

        f_iso = self.conv_iso(x).features
        f_z = self.conv_z(x).features
        f_xy = self.conv_xy(x).features

        g_emb = self.geometry_embed(geometry)
        f_emb = self.feature_embed(x.features)
        gate_input = torch.cat([g_emb, f_emb], dim=1)

        logits = self.gate_mlp(gate_input)
        weights = torch.softmax(logits, dim=1)

        fused = (
            weights[:, 0:1] * f_iso +
            weights[:, 1:2] * f_z +
            weights[:, 2:3] * f_xy)

        # Store only detached statistics to avoid retaining the graph.
        self.last_gate_mean = weights.detach().mean(dim=0)
        self.last_geometry_mean = geometry.detach().mean(dim=0)

        return x.replace_feature(fused)

    def forward(self, input):
        identity = spconv.SparseConvTensor(
            input.features,
            input.indices,
            input.spatial_shape,
            input.batch_size)

        if self.normalize_before:
            x = input.replace_feature(
                self.relu(self.pre_norm(input.features)))
            x = self._anisotropic_first_conv(x)
            x = x.replace_feature(
                self.relu(self.mid_norm(x.features)))
            x = self.conv2(x)
        else:
            x = self._anisotropic_first_conv(input)
            x = x.replace_feature(
                self.relu(self.post_norm1(x.features)))
            x = self.conv2(x)
            x = x.replace_feature(
                self.relu(self.post_norm2(x.features)))

        x = x.replace_feature(
            x.features + self.i_branch(identity).features)
        return x


@MODELS.register_module()
class SpConvUNet(nn.Module):
    """SpConv U-Net with optional GAASE blocks in selected encoder stages.

    By design, GAASE replaces ONLY block0 of the selected encoder stages.
    The second ResidualBlock, downsampling, bottleneck, decoder, skip
    connections and blocks_tail remain original.

    Encoder stage numbering:
        stage 1 -> 64 channels
        stage 2 -> 128 channels
        stage 3 -> 192 channels
        stage 4 -> 256 channels
        stage 5 -> 320 channels (bottleneck for [64,128,192,256,320])
    """

    def __init__(self,
                 num_planes,
                 norm_fn=functools.partial(
                      nn.BatchNorm1d, eps=1e-4, momentum=0.1),
                 block_reps=2,
                 block=ResidualBlock,
                 indice_key_id=1,
                 normalize_before=True,
                 return_blocks=False,
                 gaase_stages=None,
                 gaase_gate_hidden=16,
                 gaase_iso_init_bias=1.5):
        super().__init__()
        self.return_blocks = return_blocks
        self.num_planes = num_planes
        self.gaase_stages = tuple(gaase_stages or ())
        self.gaase_gate_hidden = gaase_gate_hidden
        self.gaase_iso_init_bias = gaase_iso_init_bias

        # Process block and norm_fn caller.
        if isinstance(block, str):
            area = ['residual', 'vgg', 'asym']
            assert block in area, f'block must be in {area}, but got {block}'
            if block == 'residual':
                block = ResidualBlock

        # --------------------------------------------------------------
        # Encoder same-resolution blocks.
        # Only block0 in selected stages is replaced by GAASEBlock.
        # --------------------------------------------------------------
        blocks = {}
        for i in range(block_reps):
            use_gaase = (
                indice_key_id in self.gaase_stages and i == 0)

            if use_gaase:
                blocks[f'block{i}'] = GAASEBlock(
                    num_planes[0],
                    num_planes[0],
                    norm_fn,
                    normalize_before=normalize_before,
                    indice_key=f'subm{indice_key_id}',
                    gate_hidden=gaase_gate_hidden,
                    iso_init_bias=gaase_iso_init_bias)
            else:
                blocks[f'block{i}'] = block(
                    num_planes[0],
                    num_planes[0],
                    norm_fn,
                    normalize_before=normalize_before,
                    indice_key=f'subm{indice_key_id}')

        blocks = OrderedDict(blocks)
        self.blocks = spconv.SparseSequential(blocks)

        if len(num_planes) > 1:
            # Original downsampling remains unchanged.
            if normalize_before:
                self.conv = spconv.SparseSequential(
                    norm_fn(num_planes[0]), nn.ReLU(),
                    spconv.SparseConv3d(
                        num_planes[0],
                        num_planes[1],
                        kernel_size=2,
                        stride=2,
                        bias=False,
                        indice_key=f'spconv{indice_key_id}'))
            else:
                self.conv = spconv.SparseSequential(
                    spconv.SparseConv3d(
                        num_planes[0],
                        num_planes[1],
                        kernel_size=2,
                        stride=2,
                        bias=False,
                        indice_key=f'spconv{indice_key_id}'),
                    norm_fn(num_planes[1]), nn.ReLU())

            # Recursive encoder/deeper U-Net. The same gaase_stages list is
            # passed down, so stage 1 and stage 2 are selected by indice_key_id.
            self.u = SpConvUNet(
                num_planes[1:],
                norm_fn,
                block_reps,
                block,
                indice_key_id=indice_key_id + 1,
                normalize_before=normalize_before,
                return_blocks=return_blocks,
                gaase_stages=self.gaase_stages,
                gaase_gate_hidden=gaase_gate_hidden,
                gaase_iso_init_bias=gaase_iso_init_bias)

            # Original upsampling remains unchanged.
            if normalize_before:
                self.deconv = spconv.SparseSequential(
                    norm_fn(num_planes[1]), nn.ReLU(),
                    spconv.SparseInverseConv3d(
                        num_planes[1],
                        num_planes[0],
                        kernel_size=2,
                        bias=False,
                        indice_key=f'spconv{indice_key_id}'))
            else:
                self.deconv = spconv.SparseSequential(
                    spconv.SparseInverseConv3d(
                        num_planes[1],
                        num_planes[0],
                        kernel_size=2,
                        bias=False,
                        indice_key=f'spconv{indice_key_id}'),
                    norm_fn(num_planes[0]), nn.ReLU())

            # Decoder / skip-fusion tail remains the ORIGINAL block.
            blocks_tail = {}
            for i in range(block_reps):
                blocks_tail[f'block{i}'] = block(
                    num_planes[0] * (2 - i),
                    num_planes[0],
                    norm_fn,
                    indice_key=f'subm{indice_key_id}',
                    normalize_before=normalize_before)
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = spconv.SparseSequential(blocks_tail)

    def forward(self, input, previous_outputs=None):
        output = self.blocks(input)
        identity = spconv.SparseConvTensor(
            output.features,
            output.indices,
            output.spatial_shape,
            output.batch_size)

        if len(self.num_planes) > 1:
            output_decoder = self.conv(output)
            if self.return_blocks:
                output_decoder, previous_outputs = self.u(
                    output_decoder, previous_outputs)
            else:
                output_decoder = self.u(output_decoder)
            output_decoder = self.deconv(output_decoder)

            output = output.replace_feature(
                torch.cat(
                    (identity.features, output_decoder.features), dim=1))
            output = self.blocks_tail(output)

        if self.return_blocks:
            # NOTE: to avoid the residual bug
            if previous_outputs is None:
                previous_outputs = []
            previous_outputs.append(output)
            return output, previous_outputs
        else:
            return output
