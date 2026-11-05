# Semantic segmentation ROI batch: Input (no ROI) + GT + 9 prediction models.
# Windows PowerShell 5.1 compatible. Process 11 test plants in sequence.
$ErrorActionPreference = 'Stop'

$python = 'D:\anaconda3\envs\openmmlab\python.exe'
$scriptPath = 'D:\pythoncode\oneformer3d-main\render_semantic_paper_roi.py'
$root = 'D:\pythoncode\oneformer3d-main\work_dirs'

# Use a single delimited string to avoid multiline array parser issues.
$plants = '1149-12(2);384-2(2);1149-23(2);1149-24(2);384-31(2);384-3(2);1149-13(2);1149-36(2);384-26(2);336-21(2);336-23(2)'.Split(';')

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python executable not found: $python"
}
if (-not (Test-Path -LiteralPath $scriptPath -PathType Leaf)) {
    throw "Python rendering script not found: $scriptPath"
}

foreach ($id in $plants) {
    $gt = "$root\base_vis\OneFormer3D_${id}_gt_sem.ply"
    $raw = "D:\onef\2time\${id}.ply"

    # Order: PointNet++, DGCNN, RandLA-Net, PointMLP, PMSPA-Net,
    #        SLNet, 3PNet, OneFormer3D, GAMQ-SegNet.
    $pred = @()
    $pred += "D:\pythoncode\Pointnet_Pointnet2_pytorch-master\log\sem_seg\2026-09-23_00-05\visualization_ply\PointNet++_${id}_pred_sem.ply"
    $pred += "D:\pythoncode\dgcnn.pytorch-master\outputs\dgcnn_corn_original\visualization_ply\DGCNN_${id}_pred_sem.ply"
    $pred += "D:\pythoncode\RandLA-Net-Pytorch-New-master\test_output_corn\2026-09-27_12-47-25\visualization_ply\RandLA-Net_${id}_pred_sem.ply"
    $pred += "D:\pythoncode\pointMLP-pytorch-main\part_segmentation\checkpoints\pointMLP_corn_rgb\visualization_ply\PointMLP_${id}_pred_sem.ply"
    $pred += "D:\pythoncode\PMSPA-Net-main\log\sem_seg\pmspa_formal\visualization_ply\PMSPA-Net_${id}_pred_sem.ply"
    $pred += "D:\pythoncode\SLNet-main\checkpoints\partseg_shapenet\corn_slnet_formal\visualization_ply\SLNet_${id}_pred_sem.ply"
    $pred += "D:\pythoncode\3PNet-main\logs\corn_test\visualization_ply\3PNet_${id}_pred_sem.ply"
    $pred += "$root\base_vis\OneFormer3D_${id}_pred_sem.ply"
    $pred += "$root\full_vis\GAMQ-SegNet_${id}_pred_sem.ply"

    $required = @($gt, $raw) + $pred
    $missing = @($required | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
    if ($missing.Count -gt 0) {
        Write-Host "[MISSING] ${id} (skipping)" -ForegroundColor Yellow
        $missing | ForEach-Object { Write-Host "    $_" }
        continue
    }

    $outDir = "$root\semantic_roi\$id"
    Write-Host "[PREVIEW] $id : Input + GT + 9 prediction models" -ForegroundColor Cyan

    # Argument splatting passes each path as one complete argument.
    $pythonArgs = @($scriptPath, '--reference', $gt, '--raw', $raw, '--files')
    $pythonArgs += $pred
    $pythonArgs += @('--out-dir', $outDir, '--width', '900', '--height', '1200', '--point-size', '3', '--margin', '55', '--preview-cols', '6')
    & $python @pythonArgs
    if ($LASTEXITCODE -eq 2) {
        Write-Host '[QUIT] User exited the batch.' -ForegroundColor Yellow
        break
    }
    if ($LASTEXITCODE -ne 0) {
        throw "ROI export failed for ${id}: exit code $LASTEXITCODE"
    }
}
