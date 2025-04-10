import os
import cv2
import numpy as np
import random
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from pprint import pprint
import matplotlib.pyplot as plt
import pickle

def save_preprocessed_data(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_preprocessed_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# Charger le dataset depuis Kaggle
def load_dataset(folder_path, label, cache_file, scaler=None):
    if os.path.exists(cache_file):
        print(f"→ Chargement depuis le cache : {cache_file}")
        dataset = load_preprocessed_data(cache_file)
    else:
        dataset = []
        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (64, 64))
            features = extract_hog_features(img)
            dataset.append({'features': features, 'label': label})
        print(f"Chargé {len(dataset)} images de {folder_path} avec label {label}.")
        save_preprocessed_data(dataset, cache_file)
    return dataset

# Extraction des descripteurs HOG
def extract_hog_features(image): 
    features, _ = hog(image, pixels_per_cell=(8, 8),cells_per_block=(2, 2), orientations=9, visualize=True, block_norm='L2-Hys')
    return features

# Charger les images des deux classes
vehicle_train = load_dataset("dataset_detec/train/vehicles", 1,"vehicle_train.pkl")
non_vehicle_train = load_dataset("dataset_detec/train/non_vehicles", 0,"non_vehicle_train.pkl")
dataset_train = vehicle_train + non_vehicle_train
print(f"Chargé {len(dataset_train)} images d'entraînement.")

vehicle_test = load_dataset("dataset_detec/test/vehicles", 1,"vehicle_test.pkl")
non_vehicle_test = load_dataset("dataset_detec/test/non_vehicles", 0,"non_vehicle_test.pkl")
dataset_test = vehicle_test + non_vehicle_test
print(f"Chargé {len(dataset_test)} images de test.")

vehicle_validation = load_dataset("dataset_detec/val/vehicles", 1,"vehicle_validation.pkl")
non_vehicle_validation = load_dataset("dataset_detec/val/non_vehicles", 0,"non_vehicle_validation.pkl")
dataset_validation = vehicle_validation + non_vehicle_validation
print(f"Chargé {len(dataset_validation)} images de validation.")

# Normaliser les caractéristiques
scaler = StandardScaler()
features_matrix = np.array([data['features'] for data in dataset_train])
scaled_features = scaler.fit_transform(features_matrix)

# Mettre à jour le dataset avec les caractéristiques normalisées
for i, data in enumerate(dataset_train): 
    dataset_train[i]['features'] = tuple(scaled_features[i])

X_test = np.array([d['features'] for d in dataset_test])
X_test_scaled = scaler.transform(X_test)
for i, data in enumerate(dataset_test):
    dataset_test[i]['features'] = tuple(X_test_scaled[i])
    
X_val = np.array([d['features'] for d in dataset_validation])
X_val_scaled = scaler.transform(X_val)
for i, data in enumerate(dataset_validation):
    dataset_validation[i]['features'] = tuple(X_val_scaled[i])
    
# Convertir en tuple pour l'utiliser comme clé dans la Q-Table

# Définition de l'environnement
class Environment:
    def __init__(self, dataset):
        self.dataset = dataset
        self.current_index = 0
    def reset(self):
        self.current_index = 0
        return self.dataset[self.current_index]['features']
    def step(self, action):
        reward = 1 if action == self.dataset[self.current_index]['label'] else -1
        self.current_index += 1
        done = self.current_index >= len(self.dataset)
        next_state = self.dataset[self.current_index]['features'] if not done else None
        return next_state, reward, done

# Évaluation de la performance 
def evaluate_performance(agent, dataset):
    correct = 0
    total_reward = 0
    for data in dataset:
        state = data['features']
        label = data['label']
        # si l'état n'est pas dans la Q-Table, choisir une action aléatoire
        if state in agent.q_table:
            action = np.argmax(agent.q_table[state])
        else:
            action = random.randint(0, agent.action_space - 1)
        if action == label:
            correct += 1
            total_reward += 1
        else:
            total_reward -= 1
   
    accuracy = correct / len(dataset)
    print(f"Épisode {episode+1}: Total Reward = {total_reward}, Accuracy = {accuracy:.3f}")
    return accuracy


# Agent Q-Learning
class QLearningAgent:
    def __init__(self, action_space):
        self.q_table = {}
        self.action_space = action_space
        self.learning_rate = 0.1
        self.gamma = 0.99
    def choose_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space)
            return np.argmax(self.q_table[state]) if random.random() > 0.1 else random.randint(0, self.action_space - 1)
    def update_q_value(self, state, action, reward, next_state):
        if next_state is None:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table.get(next_state,np.zeros(self.action_space)))
        self.q_table[state][action] += self.learning_rate * (target - self.q_table[state][action])

# Simulation
env = Environment(dataset_train)
agent = QLearningAgent(action_space=2)
periode_eval = 2 # Évaluer tous les 3 épisodes
eval_results = []
episodes = 10 # Nombre total d'épisodes

for episode in range(episodes):
    state = env.reset()
    done = False
    
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.update_q_value(state, action, reward, next_state)
        state = next_state

    
    if (episode + 1) % periode_eval == 0 and (episode+1)!= episodes: 
        acc = evaluate_performance(agent, dataset_test)
        eval_results.append((episode + 1, acc))
        
    if (episode + 1) == episodes:
        acc = evaluate_performance(agent, dataset_validation)
        eval_results.append((episode + 1, acc))

# Afficher la Q-Table
#print("Q-Table (partielle):")
#pprint(list(agent.q_table.items())[:3])


# Afficher les résultats d'évaluation
x_vals = [x[0] for x in eval_results]
y_vals = [x[1] for x in eval_results]

plt.plot(x_vals, y_vals, marker='o')
plt.title("Évolution de la précision au fil des épisodes")
plt.xlabel("Nombre d'épisodes")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()
