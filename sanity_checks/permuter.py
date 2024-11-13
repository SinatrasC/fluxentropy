import os
import json
import csv

def main():
    data_dir = 'experiment_data'
    files = [f for f in os.listdir(data_dir) if f.endswith('.json')]

    data_dict = {}
    for file_name in files:
        file_path = os.path.join(data_dir, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            results = data.get('results', [])
            for result in results:
                text = result.get('text')
                index = result.get('index')
                char_value = result.get('characteristic_value')
                if text not in data_dict:
                    data_dict[text] = {}
                data_dict[text][file_name] = {'Rank': index, 'Characteristic_Value': char_value}

    # Build header
    header = ['Text']
    for file_name in files:
        header.extend([f'{file_name}_Rank', f'{file_name}_Characteristic_Value'])

    # Write CSV
    with open('compiled_results.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for text, file_data in data_dict.items():
            row = [text]
            for file_name in files:
                if file_name in file_data:
                    index = file_data[file_name]['Rank']
                    char_value = file_data[file_name]['Characteristic_Value']
                    row.extend([index, char_value])
                else:
                    row.extend(['', ''])
            writer.writerow(row)

if __name__ == "__main__":
    main()
