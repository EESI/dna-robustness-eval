import pandas as pd
import subprocess
from tqdm import tqdm  

# calculate MFE with RNAfold(using vienna-rna)
def get_mfe(seq: str) -> float:
    rna_seq = seq.upper().replace('T', 'U')
    result = subprocess.run(
        ['RNAfold'],
        input=rna_seq,
        capture_output=True,
        text=True
    )
    lines = result.stdout.strip().split('\n')
    mfe_line = lines[-1]
    mfe_value = float(mfe_line.split('(')[-1].strip(')'))
    return mfe_value


