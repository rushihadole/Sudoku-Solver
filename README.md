# Sudoku-Solver

The Sudoku solver uses back-tracking algorithm to solve the Sudoku problem.
An image of problem is read using OpenCV and CNN identifies the digits.
Input to CNN is an image of Sudoku problem, it identifies the digits from the image and create the array of board to be fed to the back-tracking algorithm.
After finding the solution, it is then augmented on the image giving complete solution.
Further development will include feeding of image through webcam and augmenting the solution on the live feed video.
