def create_text_file(file_name, num_lines, values):
  """Creates a text file with the specified number of lines and values.

  Args:
    file_name: The name of the file to create.
    num_lines: The number of lines to write to the file.
    values: A list of values to be written on each line.
  """

  with open(file_name, 'w') as file:
    for _ in range(num_lines):
      file.write(', '.join(values) + '\n')

# Example usage:
create_text_file('B.txt', 500, ['0', '0', '-10', '0'])
