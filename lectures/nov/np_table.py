import numpy as np


def np_table(name, a, header=[], params=[]):
  """
    Make a simple latex table and save it to a text file

    @param: name    filename
    @param: a       numpy array
    @param: head    header string array
    @param: params  numpy array vector with used params

    @returns:
      None
  """

  assert len(a) and len(a[0])

  # dim
  n_rows = len(a)
  n_cols = len(a[0])

  # precision
  np.set_printoptions(precision=4)

  table_str = '\\begin{table}[ht!]\n\\begin{center}\n\\begin{tabular}{'

  # layout
  for col in range(n_cols):
    table_str += '| M{2cm} '

  table_str += '|}\n\\hline\n'

  # header
  if header:
    table_str += '\\rowcolor{LightYellow}\n'
    for h in header[:-1]:
      table_str += '\\textbf{' + h + '} & '

    table_str += '\\textbf{' + header[-1] + '} \\\\\n\\hline\n'

  # params
  if params:
    table_str += '\\rowcolor{LightYellow}\n'
    for p in params[:-1]:
      table_str += '\\textbf{{{:.4f}}} & '.format(p)

    table_str += '\\textbf{{{:.4f}}} \\\\\n\\hline\n'.format(params[-1])

  # content rows
  for row in range(n_rows):

    # column
    for col in range(n_cols - 1): table_str += '{} & '.format(a[row][col])
    table_str += ' {} \\\\\n'.format(a[row][col+1])

  # footer
  table_str += '\\hline\n\\end{tabular}\n\\end{center}\n\\caption{' + name +'}\n\\label{tab:' + name +'}\n\\end{table}\n\\noindent'

  #print(table_str)
  with open(name + '.txt', 'w') as f:
    f.write(table_str)