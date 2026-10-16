"""Выполнить ноутбуки текущим Python: python run_labs.py [3 4]."""
import argparse
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('labs', nargs='*', type=int, metavar='{3,4}')
    args = parser.parse_args()
    if any(number not in (3, 4) for number in args.labs):
        parser.error('Доступны только лабораторные 3 и 4')
    root = Path(__file__).resolve().parent
    for number in args.labs or [3, 4]:
        path = root / f'Lab{number}/Lab{number}.ipynb'
        notebook = nbformat.read(path, as_version=4)
        manager = KernelManager(kernel_name='python3')
        manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
        client = NotebookClient(notebook, km=manager, timeout=1800,
                                resources={'metadata': {'path': str(path.parent)}})
        print(f'Выполнение {path.name}', flush=True)
        try:
            client.execute()
            nbformat.validate(notebook)
            nbformat.write(notebook, path)
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
        print(f'Готово: {path.name}', flush=True)


if __name__ == '__main__':
    main()
