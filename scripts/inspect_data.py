"""Inspect actual Arrow schemas before selecting columns; no whole-table load."""
import argparse
from pathlib import Path
import pyarrow as pa

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('files', nargs='+', type=Path)
args = parser.parse_args()
for path in args.files:
    with pa.memory_map(str(path), 'r') as source:
        reader = pa.ipc.open_file(source)
        print(path.name, reader.schema, sep='\n')
        print('batches:', reader.num_record_batches)
        if reader.num_record_batches:
            print('sample:', reader.get_batch(0).slice(0, 3).to_pydict())
