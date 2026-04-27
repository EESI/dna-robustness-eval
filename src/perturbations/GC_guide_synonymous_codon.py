import random
import pandas as pd


codon_table = {
    'TTT': 'F', 'TTC': 'F',
    'TTA': 'L', 'TTG': 'L', 'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I',
    'ATG': 'M',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S', 'AGT': 'S', 'AGC': 'S',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'TAT': 'Y', 'TAC': 'Y',
    'CAT': 'H', 'CAC': 'H',
    'CAA': 'Q', 'CAG': 'Q',
    'AAT': 'N', 'AAC': 'N',
    'AAA': 'K', 'AAG': 'K',
    'GAT': 'D', 'GAC': 'D',
    'GAA': 'E', 'GAG': 'E',
    'TGT': 'C', 'TGC': 'C',
    'TGG': 'W',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R', 'AGA': 'R', 'AGG': 'R',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
    'TAA': '*', 'TAG': '*', 'TGA': '*'
}


aa_to_codons = {}
for codon, aa in codon_table.items():
    if aa not in aa_to_codons:
        aa_to_codons[aa] = []
    aa_to_codons[aa].append(codon)

def gc_content(seq):
    gc_count = seq.count("G") + seq.count("C")
    return gc_count / len(seq)

def gc_difference(seq1, seq2):
    return abs(gc_content(seq1) - gc_content(seq2))

def codon_synonymous_gc_guided(sequence, mutation_rate=0.1, lambda_gc=1.0):
    codons = [sequence[i:i+3] for i in range(0, len(sequence), 3)]
    original_gc = gc_content(sequence)
    for i in range(len(codons)):
        codon = codons[i]
        if len(codon) == 3 and codon in codon_table:
            if random.random() < mutation_rate:
                aa = codon_table[codon]
                synonyms = aa_to_codons[aa]
                if len(synonyms) > 1:
                    best_score = float('-inf')
                    best_codon = codon
                    for alt in synonyms:
                        if alt == codon:
                            continue
                        temp_codons = codons.copy()
                        temp_codons[i] = alt
                        mutated_seq = ''.join(temp_codons)
                        score = -lambda_gc * gc_difference(sequence, mutated_seq)
                        if score > best_score:
                            best_score = score
                            best_codon = alt
                    codons[i] = best_codon
    return ''.join(codons)

def synonymous_codon_attack_gc(sequences, mutation_rate=0.1, lambda_gc=1.0, iteration=1):
    mutated_sequences = sequences.copy()
    for _ in range(iteration):
        mutated_sequences = mutated_sequences.apply(
            lambda seq: codon_synonymous_gc_guided(seq, mutation_rate, lambda_gc)
        )
    return mutated_sequences


