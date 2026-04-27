import random
import pandas as pd



def nucleotide_mutation(sequence, mutation_rate=0.1):
    sequence = list(sequence)
    for i in range(len(sequence)):
        if random.random() < mutation_rate:
            sequence[i] = random.choice('ATCG')
    return ''.join(sequence)


def nucleotide_attack(sequences, mutation_rate=0.1, iteration=1):
  mutated_sequences = sequences.copy()  # copy original
  for _ in range(iteration):
        mutated_sequences = mutated_sequences.apply(
            lambda seq: nucleotide_mutation(seq, mutation_rate)
        )
  return mutated_sequences

