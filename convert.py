import nbformat
import os

def convert_ipynb_to_py(input_file):
    # Extract the base name without the extension
    base_name = os.path.splitext(input_file)[0]

    # Construct the output file name
    output_file = base_name + '.py'

    # Load the notebook
    with open(input_file, 'r', encoding='utf-8') as file:
        notebook = nbformat.read(file, as_version=4)

    # Open the output file
    with open(output_file, 'w', encoding='utf-8') as file:
        for cell in notebook.cells:
            # Check if the cell is a code cell
            if cell.cell_type == 'code':
                # Write the code to the output file
                file.write("#" + "-"*70 + "\n")  # Separator for readability
                file.write(cell.source + "\n\n")

    print(f'convert of {input_file} is done')

# Example usage
convert_ipynb_to_py('final v2.ipynb')