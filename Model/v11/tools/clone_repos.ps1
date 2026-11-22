$ErrorActionPreference = 'Continue'
$dst = 'C:\Users\dinos\BrainHole\Model\v11\third_party'
New-Item -ItemType Directory -Force -Path $dst | Out-Null
$env:GIT_LFS_SKIP_SMUDGE = '1'
$env:GIT_TERMINAL_PROMPT = '0'

$repos = @(
  @{ name = 'BasicVSR_PlusPlus'; url = 'https://github.com/ckkelvinchan/BasicVSR_PlusPlus.git'; role = 'flow-guided deformable alignment + grid propagation' },
  @{ name = 'RVRT';               url = 'https://github.com/JingyunLiang/RVRT.git';                role = 'recurrent VSR + guided deformable attention' },
  @{ name = 'FlashVSR';           url = 'https://github.com/OpenImagingLab/FlashVSR.git';        role = 'real-time lightweight VSR' },
  @{ name = 'deep-burst-sr';       url = 'https://github.com/goutamgmb/deep-burst-sr.git';        role = 'Deep Burst Super-Resolution (multi-frame sub-pixel align)' },
  @{ name = 'TTVSR';               url = 'https://github.com/researchmm/TTVSR.git';                role = 'test-time training VSR (offline, reference only)' },
  @{ name = 'TecoGAN';             url = 'https://github.com/thunil/TecoGAN.git';                  role = 'FRVSR/TecoGAN generative temporal consistency' },
  @{ name = 'Awesome-Super-Resolution'; url = 'https://github.com/ChaofWang/Awesome-Super-Resolution.git'; role = 'curated SR paper index' }
)

foreach ($r in $repos) {
  $target = Join-Path $dst $r.name
  if (Test-Path $target) { Write-Output ("SKIP  {0} (exists)" -f $r.name); continue }
  Write-Output ("CLONE {0} ..." -f $r.name)
  git clone --depth 1 --quiet $r.url $target 2>&1 | Out-String | Write-Output
  if ($LASTEXITCODE -eq 0) {
    $sha = (git -C $target rev-parse HEAD).Trim()
    Write-Output ("DONE  {0} sha={1}" -f $r.name, $sha)
  } else {
    Write-Output ("ERR   {0} exit={1}" -f $r.name, $LASTEXITCODE)
  }
}
Write-Output 'ALL CLONES ATTEMPTED'
