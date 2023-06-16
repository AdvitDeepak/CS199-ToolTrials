import random
import csv

from tools import MysteryTools 

def generate_data_csv(num_entries):
    with open('data_2.csv', 'w', newline='') as file:
        writer = csv.writer(file)

        mystery_tool = MysteryTools()

        pairs = []

        N = num_entries

        for _ in range(N):
            a = random.randint(-50, 50)
            b = random.randint(-50, 50)
            pairs.append((a, b))


        for i in range(num_entries):
            input_a, input_b = pairs[i]

            # Generate data for FUNC_1 (add)
            output = mystery_tool.get_func(0)(input_a, input_b)
            res = f"{input_a} {input_b} X1 {output}"
            writer.writerow(["", res])


            # Generate data for FUNC_2 (sub)
            output = mystery_tool.get_func(1)(input_a, input_b)
            res = f"{input_a} {input_b} X2 {output}"
            writer.writerow(["", res])

      
            # Generate data for FUNC_3 (mul)
            output = mystery_tool.get_func(2)(input_a, input_b)
            res = f"{input_a} {input_b} X3 {output}"
            writer.writerow(["", res])


            # Generate data for FUNC_4 (div)
            output = mystery_tool.get_func(3)(input_a, input_b)
            res = f"{input_a} {input_b} X4 {output}"
            writer.writerow(["", res])

    print("Data CSV file generated successfully.")

# Example usage: generate 100 entries for each function
generate_data_csv(10000)
