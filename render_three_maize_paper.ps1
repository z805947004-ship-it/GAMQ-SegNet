$ErrorActionPreference = 'Stop'
$python = 'D:\anaconda3\envs\openmmlab\python.exe'
$script = 'D:\pythoncode\oneformer3d-main\render_semantic_paper_v2.py'
$root = 'D:\pythoncode\oneformer3d-main\work_dirs'
$plants = @('384-2(2)', '1149-24(2)', '384-31(2)')
foreach ($id in $plants) {
    $gt = "$root\base_vis\OneFormer3D_${id}_gt_sem.ply"
    $raw = "D:\onef\2time\${id}.ply"
    $pred = @(
        "$root\base_vis\OneFormer3D_${id}_pred_sem.ply",
        "$root\full_vis\GAMQ-SegNet_${id}_pred_sem.ply",
        "D:\pythoncode\Pointnet_Pointnet2_pytorch-master\log\sem_seg\2026-09-23_00-05\visualization_ply\PointNet++_${id}_pred_sem.ply",
        "D:\pythoncode\dgcnn.pytorch-master\outputs\dgcnn_corn_original\visualization_ply\DGCNN_${id}_pred_sem.ply",
        "D:\pythoncode\RandLA-Net-Pytorch-New-master\test_output_corn\2026-09-27_12-47-25\visualization_ply\RandLA-Net_${id}_pred_sem.ply",
        "D:\pythoncode\pointMLP-pytorch-main\part_segmentation\checkpoints\pointMLP_corn_rgb\visualization_ply\PointMLP_${id}_pred_sem.ply",
        "D:\pythoncode\PMSPA-Net-main\log\sem_seg\pmspa_formal\visualization_ply\PMSPA-Net_${id}_pred_sem.ply",
        "D:\pythoncode\SLNet-main\checkpoints\partseg_shapenet\corn_slnet_formal\visualization_ply\SLNet_${id}_pred_sem.ply",
        "D:\pythoncode\3PNet-main\logs\corn_test\visualization_ply\3PNet_${id}_pred_sem.ply"
    )
    $required = @($gt, $raw) + $pred
    $missing = @($required | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
    if ($missing.Count -gt 0) {
        Write-Host "[MISSING] ${id}:" -ForegroundColor Red
        $missing | ForEach-Object { Write-Host "    $_" }
        continue
    }
    Write-Host "[RENDER] $id" -ForegroundColor Cyan
    & $python $script --reference $gt --raw $raw --files $pred --out-dir "$root\$id" --width 900 --height 1200 --point-size 3 --margin 55
    if ($LASTEXITCODE -ne 0) { throw "Rendering failed for $id with exit code $LASTEXITCODE" }
}
