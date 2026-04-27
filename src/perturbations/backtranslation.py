
import random
import pandas as pd
import numpy as np




dna_to_aa_table = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
    'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
}


aa_to_dna_table = {v: [k for k in dna_to_aa_table if dna_to_aa_table[k] == v] for v in set(dna_to_aa_table.values())}

def dna_to_aa(sequence):
    return ''.join(dna_to_aa_table.get(sequence[i:i+3], 'X') for i in range(0, len(sequence), 3))

def aa_to_dna(aa_sequence):
    return ''.join(np.random.choice(aa_to_dna_table[aa]) for aa in aa_sequence if aa in aa_to_dna_table)

def back_translation(sequence):
    aa_sequence = dna_to_aa(sequence)
    
    translated_dna = aa_to_dna(aa_sequence)
    
    return translated_dna

def backtranslation_attack(sequences, mutation_rate=None, iteration=None):
  mutated_sequences = sequences.copy()  # copy original
  for _ in range(iteration):
        mutated_sequences = mutated_sequences.apply(
            lambda seq: back_translation(seq)
        )
  return mutated_sequences
