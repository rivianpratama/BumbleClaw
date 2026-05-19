import os
import csv
import shutil
import glob

log_dir = r"D:\BumbleLog"
csv_file = os.path.join(log_dir, "scores.csv")
out_dir = os.path.join(log_dir, "face_biased_sorted")

# Clean up previously copied loose files in the out_dir
for f in glob.glob(os.path.join(out_dir, "*.*")):
    if os.path.isfile(f):
        os.remove(f)

valid_rows = []
with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        filename = row.get('screenshot')
        if not filename:
            continue
        try:
            val = row.get('face_biased', '').strip()
            score = float(val) if val else 0.0
        except ValueError:
            score = 0.0
        
        valid_rows.append((score, filename))

copied = 0
not_found = 0
for score, filename in valid_rows:
    src = os.path.join(log_dir, filename)
    if os.path.exists(src):
        # Calculate the 5-point bucket
        bin_start = int(score) // 5 * 5
        bin_end = bin_start + 4
        
        folder_name = f"score_{bin_start}-{bin_end}"
        
        target_folder = os.path.join(out_dir, folder_name)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
            
        dst = os.path.join(target_folder, filename)
        shutil.copy2(src, dst)
        copied += 1
    else:
        not_found += 1

print(f"Done. Copied {copied} files into respective score folders.")
