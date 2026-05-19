import time
import sys
import csv

def parse_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

def main():
    filepath = r"D:\BumbleLog\scores.csv"
    
    print(f"Monitoring {filepath} for realtime averages...")
    print("Press Ctrl+C to stop.")
    
    totals = {'face_biased': 0.0, 'multimodal': 0.0, 'ridge': 0.0, 'knn': 0.0}
    counts = {'face_biased': 0, 'multimodal': 0, 'ridge': 0, 'knn': 0}
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # Read header
            headers_line = f.readline()
            if not headers_line:
                print("File is empty.")
                return
                
            headers = next(csv.reader([headers_line]))
            
            try:
                fb_idx = headers.index('face_biased')
                mm_idx = headers.index('multimodal')
                rg_idx = headers.index('ridge')
                kn_idx = headers.index('knn')
            except ValueError as e:
                print(f"Header missing: {e}")
                return
            
            # Read lines continuously
            while True:
                line = f.readline()
                if not line:
                    # No new lines, wait and check again
                    time.sleep(0.5)
                    continue
                
                if not line.strip():
                    continue
                
                row = next(csv.reader([line]))
                if len(row) <= max(fb_idx, mm_idx, rg_idx, kn_idx):
                    continue
                
                # Update totals and counts
                fb = parse_float(row[fb_idx])
                if fb is not None:
                    totals['face_biased'] += fb
                    counts['face_biased'] += 1
                    
                mm = parse_float(row[mm_idx])
                if mm is not None:
                    totals['multimodal'] += mm
                    counts['multimodal'] += 1
                    
                rg = parse_float(row[rg_idx])
                if rg is not None:
                    totals['ridge'] += rg
                    counts['ridge'] += 1
                    
                kn = parse_float(row[kn_idx])
                if kn is not None:
                    totals['knn'] += kn
                    counts['knn'] += 1
                
                # Calculate averages
                avgs = {
                    'face_biased': totals['face_biased'] / counts['face_biased'] if counts['face_biased'] else 0,
                    'multimodal': totals['multimodal'] / counts['multimodal'] if counts['multimodal'] else 0,
                    'ridge': totals['ridge'] / counts['ridge'] if counts['ridge'] else 0,
                    'knn': totals['knn'] / counts['knn'] if counts['knn'] else 0
                }
                
                # Print running average on the same line
                sys.stdout.write(f"\rAvg - face_biased: {avgs['face_biased']:>7.3f} | multimodal: {avgs['multimodal']:>7.3f} | ridge: {avgs['ridge']:>7.3f} | knn: {avgs['knn']:>7.3f} | records: {counts['face_biased']:>4}")
                sys.stdout.flush()
                
    except KeyboardInterrupt:
        print("\nExiting...")
    except FileNotFoundError:
        print(f"\nError: Could not find {filepath}. Make sure the path is correct.")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()
