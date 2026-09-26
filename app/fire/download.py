"""Download a selected .pt file from the specified HF model repository.

python -m app.fire.download [--filename path/in/repo.pt]
"""
import argparse
import json
import os
import shutil
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

REPO = 'e1250/safety_detection'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--filename', help='模型仓库内的 .pt 文件路径；多份权重时必须指定')
    args = parser.parse_args()
    endpoint = os.environ.get('HF_ENDPOINT', 'https://huggingface.co').rstrip('/')
    with urlopen(f'{endpoint}/api/models/{REPO}', timeout=30) as response:
        info = json.load(response)
    files = [item['rfilename'] for item in info.get('siblings', []) if item['rfilename'].endswith('.pt')]
    filename = args.filename
    if filename is None and len(files) == 1:
        filename = files[0]
    if filename not in files:
        parser.error(f'请用 --filename 选择模型权重，可用文件：{files}')
    revision = info['sha']
    target = Path(__file__).resolve().parents[2] / 'models/fire/model.pt'
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        parser.error(f'{target} 已存在；请先另行保存旧权重再下载')
    # Resolve an immutable revision so metadata and downloaded bytes match.
    url = f'{endpoint}/{REPO}/resolve/{revision}/{quote(filename, safe="/")}'
    temporary = target.with_suffix('.pt.part')
    try:
        with urlopen(url, timeout=120) as response, temporary.open('wb') as output:
            shutil.copyfileobj(response, output)
        if temporary.stat().st_size < 1024:
            raise RuntimeError('下载结果过小，不是有效的模型文件')
        temporary.replace(target)
        target.with_suffix('.json').write_text(json.dumps({
            'repo': REPO, 'revision': revision, 'filename': filename,
        }, ensure_ascii=False, indent=2))
    finally:
        temporary.unlink(missing_ok=True)
    print(f'已下载 {REPO}@{revision}/{filename} -> {target}')


if __name__ == '__main__':
    main()
