import numpy as np

all_predictions_np = np.array(all_predictions_original)
adv_predictions_np = np.array(all_predictions_adv)
all_true_labels_np = np.array(all_true_labels)

correct_mask = all_predictions_np == all_true_labels_np 
changed_mask = all_predictions_np != adv_predictions_np

# measure success rate (correct → change ratio)
attack_success_rate = (np.sum(correct_mask & changed_mask) / np.sum(correct_mask)) * 100
print(f"Adversarial Attack Success Rate: {attack_success_rate:.2f}%")
