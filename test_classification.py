"""Test disease classification for all 3 classes"""
import requests
import os
import glob

url = 'http://localhost:5000/api/predict'
data = {'location_id': 1, 'season': 'Maha', 'temperature': 28.0, 'rainfall': 1500}

# Test Rice Blast (Leafsmut folder)
blast_files = sorted(glob.glob('rice leaf diseases dataset/Leafsmut/BLAST1_*.jpg'))[:20]
print(f"Testing {len(blast_files)} Rice Blast images...")
correct = 0
wrong = 0
for img_path in blast_files:
    with open(img_path, 'rb') as f:
        files = {'image': f}
        r = requests.post(url, files=files, data=data)
    result = r.json()
    det = result['disease_detection']
    detected = det['disease_name']
    conf = det['confidence']
    bb = det['all_predictions']['Bacterial Blight']
    bs = det['all_predictions']['Brown Spot']
    rb = det['all_predictions']['Rice Blast']
    
    status = 'OK' if detected == 'Rice Blast' else 'WRONG'
    if detected == 'Rice Blast':
        correct += 1
    else:
        wrong += 1
    print(f"  [{status}] {os.path.basename(img_path)} -> {detected} ({conf}%) | BB:{bb}% BS:{bs}% RB:{rb}%")

print(f"\nRice Blast: {correct}/{len(blast_files)} correct ({correct/len(blast_files)*100:.1f}%)")

# Test Bacterial Blight
bb_files = sorted(glob.glob('rice leaf diseases dataset/Bacterialblight/BACTERAILBLIGHT3_*.jpg'))[:10]
print(f"\nTesting {len(bb_files)} Bacterial Blight images...")
bb_correct = 0
for img_path in bb_files:
    with open(img_path, 'rb') as f:
        files = {'image': f}
        r = requests.post(url, files=files, data=data)
    result = r.json()
    detected = result['disease_detection']['disease_name']
    conf = result['disease_detection']['confidence']
    status = 'OK' if detected == 'Bacterial Blight' else 'WRONG'
    if detected == 'Bacterial Blight':
        bb_correct += 1
    print(f"  [{status}] {os.path.basename(img_path)} -> {detected} ({conf}%)")
print(f"Bacterial Blight: {bb_correct}/{len(bb_files)} correct")

# Test Brown Spot
bs_files = sorted(glob.glob('rice leaf diseases dataset/Brownspot/BROWNSPOT1_*.jpg'))[:10]
print(f"\nTesting {len(bs_files)} Brown Spot images...")
bs_correct = 0
for img_path in bs_files:
    with open(img_path, 'rb') as f:
        files = {'image': f}
        r = requests.post(url, files=files, data=data)
    result = r.json()
    detected = result['disease_detection']['disease_name']
    conf = result['disease_detection']['confidence']
    status = 'OK' if detected == 'Brown Spot' else 'WRONG'
    if detected == 'Brown Spot':
        bs_correct += 1
    print(f"  [{status}] {os.path.basename(img_path)} -> {detected} ({conf}%)")
print(f"Brown Spot: {bs_correct}/{len(bs_files)} correct")
