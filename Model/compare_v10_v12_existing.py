"""Compare existing v10/v12 block metrics and extract aligned video frames."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess

from record_experiment import digest, run_logged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Use a new output directory')
    args.out.mkdir(parents=True)
    runs = Path(__file__).parent / 'runs'
    folders = {
        'v10': runs / 'eval-v10-b-scene-detail-best-seed0-20261122',
        'v12': runs / 'v12-bbb-q5-phase-decay02-20261122',
    }
    rows = {name: {(r['frame'], r['block']): r for r in
                  map(json.loads, (folder / 'blocks.jsonl').read_text().splitlines())}
            for name, folder in folders.items()}
    keys = sorted(rows['v10'].keys() & rows['v12'].keys())
    assert len(keys) == 450 and keys[0] == (736, 0) and keys[-1] == (750, 29)
    methods = {'v10-gamma0.5': ('v10', 'v10-gamma0.5'),
               'v10-gamma1': ('v10', 'v10-gamma1'),
               'v10-B': ('v10', 'frozen-b-anchor'), 'v12': ('v12', 'v12')}
    metrics = {label: {metric: statistics.mean(rows[name][k][metric][method] for k in keys)
                       for metric in ('raw_projection_mae', 'rgb_reference_mae')}
               for label, (name, method) in methods.items()}
    report = {'operation': 'aggregate existing block metrics; extract existing H264 frames; no inference',
              'samples': len(keys), 'first_frame': 736, 'last_frame': 750,
              'metrics': metrics, 'inputs': {}, 'commands': [],
              'limitations': '15 frames of same-video scene only; v10 uses two frames and 60 epochs on 390 training strips, best epoch40 selected on 128 validation strips; v12 uses 16 frames and no training. RGB reference is demosaiced target RAW, not original RGB. Still frames do not establish motion quality.'}
    (args.out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for name, folder in folders.items():
        report['inputs'][name] = {str(folder / file): digest(folder / file)
                                  for file in ('blocks.jsonl', 'report.json', 'generated.mp4')}
        for frame in (736, 743, 750):
            command = ['ffmpeg', '-v', 'error', '-n', '-i', str(folder / 'generated.mp4'),
                       '-vf', f'select=eq(n\\,{frame-721})', '-frames:v', '1',
                       str(args.out / f'{name}-frame-{frame}.png')]
            report['commands'].append(command)
            subprocess.run(command, check=True, capture_output=True)
    report['outputs'] = {p.name: digest(p) for p in args.out.glob('*.png')}
    (args.out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'samples': len(keys), 'metrics': metrics}, indent=2))


if __name__ == '__main__':
    run_logged('evaluation', main)
