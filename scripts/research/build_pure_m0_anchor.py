"""Convert the frozen untrained legacy anchor to pure M0, without constructing KG.

python -I -B scripts/research/build_pure_m0_anchor.py
Outputs are exclusive-create files under artifacts/mstca_ablation_factory_v1_20261106.
This historical compatibility artifact is NOT used by module_seed_v1 ablations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'packages/uav_fault'), str(ROOT/'scripts/research')]
import torch
from train_source_normal_sampling import state_digest
from uav_fault.models.cnn.ablation_factory import PURE_M0_DIGEST, build_variant

OUTPUT_ROOT = ROOT/'artifacts/mstca_ablation_factory_v1_20261106'
DEFAULT_PROTOCOL = ROOT/'configs/experiments/causal_pure_mstca_m0_v1.json'
DEFAULT_OUTPUT = OUTPUT_ROOT/'pure_anchor'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def convert_anchor(protocol_path=DEFAULT_PROTOCOL, output_dir=DEFAULT_OUTPUT):
    protocol_path, output_dir = Path(protocol_path), Path(output_dir).resolve()
    if not output_dir.is_relative_to(OUTPUT_ROOT.resolve()) or output_dir == OUTPUT_ROOT.resolve():
        raise ValueError('Use a fresh output subdirectory under the audit artifact root')
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {output_dir}')
    lock = protocol_path.with_suffix(protocol_path.suffix+'.sha256')
    if sha256(protocol_path) != lock.read_text(encoding='utf-8').strip():
        raise ValueError('Frozen protocol SHA mismatch')
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    spec = protocol['initialization_anchor']
    source_path = ROOT/spec['source_path_for_preparation']
    original_sha = sha256(source_path)
    if original_sha != spec['sha256']:
        raise ValueError('Source anchor SHA mismatch')
    source = torch.load(source_path, map_location='cpu', weights_only=True)
    if not isinstance(source, dict) or not all(isinstance(k, str) and isinstance(v, torch.Tensor)
                                               for k,v in source.items()):
        raise ValueError('Source anchor must be a tensor state dictionary')
    original_digest = state_digest(source)
    if original_digest != spec['state_digest']:
        raise ValueError('Source anchor state digest mismatch')
    template = build_variant('mstca').state_dict()
    mapping, pure = {}, {}
    for target, expected in template.items():
        origin = 'raw_encoder.'+target[len('encoder.'):] if target.startswith('encoder.') else target
        value = source.get(origin)
        if not isinstance(value, torch.Tensor) or value.shape != expected.shape or value.dtype != expected.dtype:
            raise ValueError('Common anchor tensor layout mismatch: '+target)
        mapping[target] = origin
        pure[target] = value.detach().cpu().clone()
    digest = state_digest(pure)
    if len(pure) != 41 or digest != PURE_M0_DIGEST or digest != spec['m0_mapped_state_digest']:
        raise ValueError('Pure M0 mapped state digest/count mismatch')
    # All input validation precedes output creation. Original files are never written.
    output_dir.mkdir(parents=True, exist_ok=False)
    target = output_dir/'pure_m0_seed42.pt'
    with target.open('xb') as stream:
        torch.save(pure, stream)
    receipt = {'purpose': 'historical_seed42_compatibility_only_not_module_seed_v1',
               'source_path': str(source_path), 'source_sha256': original_sha,
               'source_state_digest': original_digest, 'mapping': mapping,
               'tensor_count': len(pure), 'new_file': str(target),
               'new_file_sha256': sha256(target), 'pure_state_digest': digest,
               'protocol_sha256': sha256(protocol_path), 'torch': str(torch.__version__)}
    with (output_dir/'receipt.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(convert_anchor(args.protocol, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
