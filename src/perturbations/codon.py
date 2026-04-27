import random
import pandas as pd



def codon_mutation(sequence, mutation_rate=0.1):
    codons = [sequence[i:i+3] for i in range(0, len(sequence), 3)]
    for i in range(len(codons)):
        if random.random() < mutation_rate:
            codons[i] = ''.join(random.choices('ATCG', k=3))
    return ''.join(codons)


def codon_attack(sequences, mutation_rate=0.1, iteration=1):
  mutated_sequences = sequences.copy()  # copy original
  for _ in range(iteration):
        mutated_sequences = mutated_sequences.apply(
            lambda seq: codon_mutation(seq, mutation_rate)
        )
  return mutated_sequences
