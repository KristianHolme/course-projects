import sympy as sp
import numpy as np
import argparse

h = sp.symbols('h')
def Cmat(p0, M):
  C = np.zeros((M+1, M+1), dtype=object)
  for j, p in enumerate(range(p0, p0+M+1)):
    for i in range(M+1):
      C[j, i] = (p*h)**i / sp.factorial(i) 
  return sp.Matrix(C)


def main():
    parser = argparse.ArgumentParser(description="Calculate the C matrix.")
    parser.add_argument("p0", type=int, help="Initial power p0")
    parser.add_argument("M", type=int, help="Size of the matrix M")
    
    args = parser.parse_args()

    # Call the Cmat function with the parsed arguments
    result_matrix = Cmat(args.p0, args.M)

    # Print the result
    print("Coeffs Matrix C^-1:")
    sp.pprint(result_matrix.inv())

if __name__ == "__main__":
    main()