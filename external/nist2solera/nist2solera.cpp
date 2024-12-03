/**
 * @file adp.cpp
 * @author J. M. Recio-Lopez (jrecio1@us.es)
 * @brief This file realizes the conversion of a file from LAMMPS format to
 * SOLERA format.
 * @note This file is only verified for N=2 elements, although it has been
 * created for N>=1.
 * @version 0.1
 * @date 2024-07-31
 *
 * @copyright Copyright (c) 2024
 */

#include <array>     // Include array library for fixed-size arrays
#include <cmath>     // Include math library for mathematical functions
#include <iomanip>   // Include input/output manipulator library
#include <iostream>  // Include C++ input/output stream library
#include <stdio.h>   // Include standard input/output library
#include <string.h>  // Include library to manipulate string
#include <vector>    // Include vector library for dynamic arrays

#define LIM 500  // Define a constant LIM with value 500
#define TOL \
  0.0001  // Define a constant TOL with value 10^(-4).
          // This constant is used to add the cutoff point or not add to the
          // potential functions.

using namespace std;         // Use the standard namespace
using vec = vector<double>;  // Define an alias 'vec' for vector<double>

//*****************************************************************************

// Define a structure to store chemical element information
struct Element {
  char* name;     // Element name
  int Z;          // Atomic number
  double M;       // Atomic mass
  double a;       // Lattice parameter
  char* lattice;  // Crystal lattice type
};

// Define a structure to store cubic spline coefficients
struct SplineSet {
  double a;  // Coefficient a
  double b;  // Coefficient b
  double c;  // Coefficient c
  double d;  // Coefficient d
  double x;  // x value
};

// Definition of the function to calculate natural cubic splines:
// spline_i(x)=a+b*(x-x_i)+c*(x-x_i)^2+d*(x-x_i)^3, i=segment of the spline
vector<SplineSet> natural_cubic_splines(vec& x, vec& y) {
  int N = x.size();  // Number of points
  int n = N - 1;     // Number of segments
  vec a(n);          // Vector for a coefficients
  a = y;             // Insert y values into a
  vec b(n);          // Vector for b coefficients
  vec d(n);          // Vector for d coefficients
  vec h(n);          // Vector for x differences
  int i;             // Declare an iteration variable
  for (i = 0; i < n; i++) {
    h[i] = x[i + 1] - x[i];  // Calculate differences between consecutive x's
  }
  vec alpha(n);  // Vector for alpha values
  alpha[0] = 0;  // Add initial 0 to alpha
  for (i = 1; i < n; i++) {
    alpha[i] = (3 * (a[i + 1] - a[i]) / h[i] -
                3 * (a[i] - a[i - 1]) / h[i - 1]);  // Calculate alpha values
  }
  vec c(n + 1);   // Vector for c coefficients
  vec l(n + 1);   // Vector for l values
  vec mu(n + 1);  // Vector for mu values
  vec z(n + 1);   // Vector for z values
  l[0] = 1;
  mu[0] = 0;
  z[0] = 0;
  for (i = 1; i < n; i++) {
    l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1];
    mu[i] = h[i] / l[i];
    z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
  }

  l[n] = 1;
  z[n] = 0;
  c[n] = 0;
  for (int j = n - 1; j >= 0; j = j - 1) {
    c[j] = z[j] - mu[j] * c[j + 1];
    b[j] = (a[j + 1] - a[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3;
    d[j] = (c[j + 1] - c[j]) / 3 / h[j];
  }

  // Define the output vector of SplineSet structures
  vector<SplineSet> output_set(n);
  for (int i = 0; i < n; i++) {
    output_set[i].a = a[i];
    output_set[i].b = b[i];
    output_set[i].c = c[i];
    output_set[i].d = d[i];
    output_set[i].x = x[i];
  }
  return output_set;
}

//*****************************************************************************
// Main function: reads input file, processes data, and writes output files
//*****************************************************************************
int main(int argc, char** argv) {

  // Check if the filename argument is provided
  if (argc < 2 || argv[1] == nullptr || argv[1][0] == '\0') {
    throw std::invalid_argument("Invalid filename argument.");
  }

  char filename[100];
  snprintf(filename, sizeof(filename), "%s", argv[1]);

  std::cout << "Reading file: " << filename << std::endl;

  FILE* file;                   // Declare a file pointer
  file = fopen(filename, "r");  // Open "filename.adp" file in read mode
  char line[LIM];               // Declare a character array to store file lines
  fgets(line, LIM, file);       // Read the first line of the file
  fgets(line, LIM, file);       // Read the second line of the file
  fgets(line, LIM, file);       // Read the third line of the file
  int N;                        // Declare a variable for the number of elements
  fscanf(file, "%i", &N);       // Read the number of elements from the file
  int comb;                // Declare a variable for the number of combinations
  comb = N * (N + 1) / 2;  // Calculate the number of combinations
  int i;                   // Declare an iteration variable
  struct Element Element[N];  // Declare an array of Element structures
  char e[N][2 + 1];           // Declare a 2D array to store element symbols

  for (i = 0; i < N; i++) {
    fscanf(file, "%s", e[i]);  // Read the element symbol from the file
    Element[i].name = e[i];    // Assign the read symbol to the element name
  }

  int N_rho;     // Declare variable for number of density points
  int N_r;       // Declare variable for number of distance points
  double d_rho;  // Declare variable for density step
  double d_r;    // Declare variable for distance step
  double r_c;    // Declare variable for cutoff radius

  fscanf(file, "%i", &N_rho);   // Read N_rho from file
  fscanf(file, "%lf", &d_rho);  // Read d_rho from file
  fscanf(file, "%i", &N_r);     // Read N_r from file
  fscanf(file, "%lf", &d_r);    // Read d_r from file
  fscanf(file, "%lf", &r_c);    // Read r_c from file

  int cols;
  int cond;
  if (r_c - (N_r - 1) * d_r < TOL) {
    cond = 0;  // not add cutoff point
    cols = N_r;
  } else {
    cond = 1;  // add cutoff point
    cols = N_r + 1;
  }
  int j;                      // Declare another iteration variable
  char lattice[N][LIM];       // Declare 2D array for crystal lattice types
  double F_rho[N][N_rho];     // Declare 2D array for embedding function
  double rho_r[N][cols];      // Declare 2D array for electron density
  double rphi_r[N][N][cols];  // Declare 3D array for pair potential
  double u_r[N][N][cols];     // Declare 3D array for function u(r)
  double w_r[N][N][cols];     // Declare 3D array for function w(r)

  for (i = 0; i < N; i++) {
    fscanf(file, "%i", &Element[i].Z);   // Read atomic number
    fscanf(file, "%lf", &Element[i].M);  // Read atomic mass
    fscanf(file, "%lf", &Element[i].a);  // Read lattice parameter
    fscanf(file, "%s", lattice[i]);      // Read crystal lattice type
    Element[i].lattice = lattice[i];     // Assign lattice type to element
    cout << "Z=" << Element[i].Z << " M=" << Element[i].M
         << " a=" << Element[i].a << " lattice=" << Element[i].lattice
         << " name=" << Element[i].name << endl;
    for (j = 0; j < N_rho; j++) {
      fscanf(file, "%lf", &F_rho[i][j]);  // Read F_rho values
    }
    for (j = 0; j < N_r; j++) {
      fscanf(file, "%lf", &rho_r[i][j]);  // Read rho_r values
    }
    if (cond == 1) rho_r[i][N_r] = 0;  // Set the last rho_r value to 0
  }

  int k;  // Declare another iteration variable
  for (j = 0; j < N; j++) {
    for (i = 0; i <= j; i++) {
      for (k = 0; k < N_r; k++) {
        fscanf(file, "%lf", &rphi_r[i][j][k]);  // Read rphi_r values
      }
      if (cond == 1) rphi_r[i][j][N_r] = 0;  // Set the last rphi_r value to 0
    }
  }
  for (j = 0; j < N; j++) {
    for (i = 0; i <= j; i++) {
      for (k = 0; k < N_r; k++) {
        fscanf(file, "%lf", &u_r[i][j][k]);  // Read u_r values
      }
      if (cond == 1) u_r[i][j][N_r] = 0;  // Set the last u_r value to 0
    }
  }
  for (j = 0; j < N; j++) {
    for (i = 0; i <= j; i++) {
      for (k = 0; k < N_r; k++) {
        fscanf(file, "%lf", &w_r[i][j][k]);  // Read w_r values
      }
      if (cond == 1) w_r[i][j][N_r] = 0;  // Set the last w_r value to 0
    }
  }
  fclose(file);  // Close the input file

  // Scaled data conversion of the function phi(r)
  double r[cols];
  double rho[N_rho];
  double phi_r[N][N][cols];

  for (j = 0; j < N; j++) {
    for (i = 0; i <= j; i++) {
      for (k = 1; k < cols; k++) {
        r[k] = k * d_r;                           // Calculate r values
        phi_r[i][j][k] = rphi_r[i][j][k] / r[k];  // Calculate phi_r values
      }
      // Calculate phi_r value for r=0 using cubic polynomial extrapolation.
      r[0] = 0;
      phi_r[i][j][0] = (4 * phi_r[i][j][1] - 6 * phi_r[i][j][2] +
                        4 * phi_r[i][j][3] - phi_r[i][j][4]);
    }
  }

 // Calculate rho values
  for (j = 0; j < N_rho; j++) {
    rho[j] = j * d_rho;                        
  }

  ///////////////////////////////////////////////////////////////////////
  // Save data
  ///////////////////////////////////////////////////////////////////////
#ifdef DEBUG_MODE
  FILE* F_file;
  FILE* rho_file;
  FILE* phi_file;
  FILE* u_file;
  FILE* w_file;

  F_file = fopen("F.dat", "w");  // Open F.dat file for writing
  for (j = 0; j < N_rho; j++) {
    fprintf(F_file, "%i %1.16lf ", j, rho[j]);  // Write rho values
    for (i = 0; i < N; i++) {
      fprintf(F_file, "%1.16lf ", F_rho[i][j]);  // Write F_rho values
    }
    fprintf(F_file, "\n");
  }
  fclose(F_file);  // Close F.dat file
  printf("File created: F.dat \n");

  rho_file = fopen("rho.dat", "w");  // Open rho.dat file for writing
  for (j = 0; j < cols; j++) {
    fprintf(rho_file, "%i %1.16lf ", j, r[j]);  // Write r values
    for (i = 0; i < N; i++) {
      fprintf(rho_file, "%1.16lf ", rho_r[i][j]);  // Write rho_r values
    }
    fprintf(rho_file, "\n");
  }
  fclose(rho_file);  // Close rho.dat file
  printf("File created: rho.dat \n");

  phi_file = fopen("phi.dat", "w");  // Open phi.dat file for writing
  for (k = 0; k < cols; k++) {
    fprintf(phi_file, "%i %1.16lf ", k, r[k]);  // Write r values
    for (j = 0; j < N; j++) {
      for (i = 0; i <= j; i++) {
        fprintf(phi_file, "%1.16lf ", phi_r[i][j][k]);  // Write phi_r values
      }
    }
    fprintf(phi_file, " \n");
  }
  fclose(phi_file);  // Close phi.dat file
  printf("File created: phi.dat \n");

  u_file = fopen("u.dat", "w");  // Open u.dat file for writing
  for (k = 0; k < cols; k++) {
    fprintf(u_file, "%i %1.16lf ", k, r[k]);  // Write r values
    for (j = 0; j < N; j++) {
      for (i = 0; i <= j; i++) {
        fprintf(u_file, "%1.16lf ", u_r[i][j][k]);  // Write u_r values
      }
    }
    fprintf(u_file, " \n");
  }
  fclose(u_file);  // Close u.dat file
  printf("File created: u.dat \n");

  w_file = fopen("w.dat", "w");  // Open w.dat file for writing
  for (k = 0; k < cols; k++) {
    fprintf(w_file, "%i %1.16lf ", k, r[k]);  // Write r values
    for (j = 0; j < N; j++) {
      for (i = 0; i <= j; i++) {
        fprintf(w_file, "%1.16lf ", w_r[i][j][k]);  // Write w_r values
      }
    }
    fprintf(w_file, " \n");
  }
  fclose(w_file);  // Close w.dat file
  printf("File created: w.dat \n");
#endif

  ///////////////////////////////////////////////////////////
  // Create files with spline data to SOLERA format
  ///////////////////////////////////////////////////////////
  for (j = 0; j < N; j++) {
    for (i = 0; i <= j; i++) {
      if (i == j) {
        FILE* Spline_file;
        char prefix[LIM] = "adp";
        char ext[LIM] = ".dat";
        char* filename;
        filename = strcat(prefix, e[i]);
        filename = strcat(filename, ext);
        Spline_file = fopen(filename, "w+");

        // Print the header
        fprintf(Spline_file, "%1.5f ",
                Element[i].M);  // Write the atomic mass M
        double factor = 1.0;
        fprintf(Spline_file, "%1.1lf ", factor);  // Write the value of factor*/
        fprintf(Spline_file, "%1.16lf \n", r_c);  // Write the value of r_cutoff

        // function rho_r
        int n = cols - 1;  // number of segments n
        vec x(n + 1);
        vec y(n + 1);
        fprintf(Spline_file, "%i ", n);  // Write the number of segments
        fprintf(Spline_file, "%1.16lf\n",
                d_r);  // Write the value of increment in r
        int l;
        for (l = 0; l <= n; l = l + 1) {

          x[l] = r[l];         // Set x values for spline calculation
          y[l] = rho_r[i][l];  // Set y values for spline calculation
        }

        vector<SplineSet> cs =
            natural_cubic_splines(x, y);  // Calculate cubic splines
        for (l = 0; l < n; l = l + 1) {
          fprintf(Spline_file,
                  "%1.16lf %1.16lf %1.16lf %1.16lf %1.16lf %1.16lf %1.16lf "
                  "%1.16lf %1.16lf",
                  cs[l].a, cs[l].b, cs[l].c, cs[l].d,  //! a, b, c, and d
                  cs[l].b, 2 * cs[l].c, 3 * cs[l].d,   //! db, dc, and dd
                  2 * cs[l].c, 6 * cs[l].d);           //! ddc and ddd
          fprintf(Spline_file, "\n");
        }

        // function F_rho
        int nn = N_rho - 1;
        vec xx(nn + 1);
        vec yy(nn + 1);
        fprintf(Spline_file, "%i ", nn);  // Write the number of segments
        fprintf(Spline_file, "%1.16lf\n",
                d_rho);  // Write the value of increment in rho
        for (l = 0; l <= nn; l = l + 1) {
          xx[l] = rho[l];       // Set x values for spline calculation
          yy[l] = F_rho[i][l];  // Set y values for spline calculation
        }
        cs = natural_cubic_splines(xx, yy);  // Calculate cubic splines
        for (l = 0; l < nn; l = l + 1) {
          fprintf(Spline_file,
                  "%1.16lf %1.16lf %1.16lf %1.16lf %1.16lf %1.16lf %1.16lf "
                  "%1.16lf %1.16lf",
                  cs[l].a, cs[l].b, cs[l].c, cs[l].d,  //! a, b, c, and d
                  cs[l].b, 2 * cs[l].c, 3 * cs[l].d,   //! db, dc, and dd
                  2 * cs[l].c, 6 * cs[l].d);           //! ddc and ddd
          fprintf(Spline_file, "\n");
        }

        // function phi_r
        fprintf(Spline_file, "%i ", n);  // Write the number of segments
        fprintf(Spline_file, "%1.16lf\n",
                d_r);  // Write the value of increment in r
        for (l = 0; l <= n; l = l + 1) {
          x[l] = r[l];            // Set x values for spline calculation
          y[l] = phi_r[i][j][l];  // Set y values for spline calculation
        }
        cs = natural_cubic_splines(x, y);  // Calculate cubic splines
        for (l = 0; l < n; l = l + 1) {
          fprintf(Spline_file,
                  "%1.16lf %1.16lf %1.16lf %1.16lf %1.16lf %1.16lf %1.16lf "
                  "%1.16lf %1.16lf",
                  cs[l].a, cs[l].b, cs[l].c, cs[l].d,  //! a, b, c, and d
                  cs[l].b, 2 * cs[l].c, 3 * cs[l].d,   //! db, dc, and dd
                  2 * cs[l].c, 6 * cs[l].d);           //! ddc and ddd
          fprintf(Spline_file, "\n");
        }

        // function u_r
        fprintf(Spline_file, "%i ", n);  // Write the number of segments
        fprintf(Spline_file, "%1.16lf\n",
                d_r);  // Write the value of increment in r
        for (l = 0; l <= n; l = l + 1) {
          x[l] = r[l];          // Set x values for spline calculation
          y[l] = u_r[i][j][l];  // Set y values for spline calculation
        }
        cs = natural_cubic_splines(x, y);  // Calculate cubic splines
        for (l = 0; l < n; l = l + 1) {
          fprintf(Spline_file,
                  "%1.16lf %1.16lf %1.16lf %1.16lf %1.16lf %1.16lf %1.16lf "
                  "%1.16lf %1.16lf",
                  cs[l].a, cs[l].b, cs[l].c, cs[l].d,  //! a, b, c, and d
                  cs[l].b, 2 * cs[l].c, 3 * cs[l].d,   //! db, dc, and dd
                  2 * cs[l].c, 6 * cs[l].d);           //! ddc and ddd
          fprintf(Spline_file, "\n");
        }

        // function w_r
        fprintf(Spline_file, "%i ", n);  // Write the number of segments
        fprintf(Spline_file, "%1.16lf\n",
                d_r);  // Write the value of increment in r
        for (l = 0; l <= n; l = l + 1) {
          x[l] = r[l];          // Set x values for spline calculation
          y[l] = w_r[i][j][l];  // Set y values for spline calculation
        }
        cs = natural_cubic_splines(x, y);  // Calculate cubic splines
        for (l = 0; l < n; l = l + 1) {
          fprintf(Spline_file,
                  "%1.16lf %1.16lf %1.16lf %1.16lf %1.16lf %1.16lf %1.16lf "
                  "%1.16lf %1.16lf",
                  cs[l].a, cs[l].b, cs[l].c, cs[l].d,  //! a, b, c, and d
                  cs[l].b, 2 * cs[l].c, 3 * cs[l].d,   //! db, dc, and dd
                  2 * cs[l].c, 6 * cs[l].d);           //! ddc and ddd
          fprintf(Spline_file, "\n");
        }
        // Close Spline_data.txt file
        fclose(Spline_file);
        printf("File created: %s \n", filename);
      } else {
        FILE* Spline_file;
        char prefix[LIM] = "adp";
        char ext[LIM] = ".dat";
        char* filename;
        filename = strcat(prefix, e[i]);
        filename = strcat(filename, e[j]);
        filename = strcat(filename, ext);
        Spline_file = fopen(filename, "w");

        // Print the header
        // fprintf(Spline_file,"%1.16lf ",Element[i].M);  // Write the atomic
        // number Z double factor=1.0; fprintf(Spline_file,"%1.1lf ",factor); //
        // Write the value of factor
        fprintf(Spline_file, "%1.16lf\n", r_c);  // Write the value of r_cutoff

        // function phi_r
        int n = cols - 1;  // number of segments n
        vec x(n + 1);
        vec y(n + 1);
        fprintf(Spline_file, "%i ", n);  // Write the number of segments
        fprintf(Spline_file, "%1.16lf\n",
                d_r);  // Write the value of increment in r
        int l;
        for (l = 0; l <= n; l = l + 1) {
          x[l] = r[l];            // Set x values for spline calculation
          y[l] = phi_r[i][j][l];  // Set y values for spline calculation
        }
        vector<SplineSet> cs =
            natural_cubic_splines(x, y);  // Calculate cubic splines
        for (l = 0; l < n; l = l + 1) {
          fprintf(Spline_file,
                  "%1.16lf %1.16lf %1.16lf %1.16lf %1.16lf %1.16lf %1.16lf "
                  "%1.16lf %1.16lf",
                  cs[l].a, cs[l].b, cs[l].c, cs[l].d,  //! a, b, c, and d
                  cs[l].b, 2 * cs[l].c, 3 * cs[l].d,   //! db, dc, and dd
                  2 * cs[l].c, 6 * cs[l].d);           //! ddc and ddd
          fprintf(Spline_file, "\n");
        }

        // function u_r
        fprintf(Spline_file, "%i ", n);  // Write the number of segments
        fprintf(Spline_file, "%1.16lf\n",
                d_r);  // Write the value of increment in r
        for (l = 0; l <= n; l = l + 1) {
          x[l] = r[l];          // Set x values for spline calculation
          y[l] = u_r[i][j][l];  // Set y values for spline calculation
        }
        cs = natural_cubic_splines(x, y);  // Calculate cubic splines
        for (l = 0; l < n; l = l + 1) {
          fprintf(Spline_file,
                  "%1.16lf %1.16lf %1.16lf %1.16lf %1.16lf %1.16lf %1.16lf "
                  "%1.16lf %1.16lf",
                  cs[l].a, cs[l].b, cs[l].c, cs[l].d,  //! a, b, c, and d
                  cs[l].b, 2 * cs[l].c, 3 * cs[l].d,   //! db, dc, and dd
                  2 * cs[l].c, 6 * cs[l].d);           //! ddc and ddd
          fprintf(Spline_file, "\n");
        }
        // function w_r
        fprintf(Spline_file, "%i ", n);  // Write the number of segments
        fprintf(Spline_file, "%1.16lf\n",
                d_r);  // Write the value of increment in r (in Amstrong)
        for (l = 0; l <= n; l = l + 1) {
          x[l] = r[l];          // Set x values for spline calculation
          y[l] = w_r[i][j][l];  // Set y values for spline calculation
        }
        cs = natural_cubic_splines(x, y);  // Calculate cubic splines
        for (l = 0; l < n; l = l + 1) {
          fprintf(Spline_file,
                  "%1.16lf %1.16lf %1.16lf %1.16lf %1.16lf %1.16lf %1.16lf "
                  "%1.16lf %1.16lf",
                  cs[l].a, cs[l].b, cs[l].c, cs[l].d,  //! a, b, c, and d
                  cs[l].b, 2 * cs[l].c, 3 * cs[l].d,   //! db, dc, and dd
                  2 * cs[l].c, 6 * cs[l].d);           //! ddc and ddd
          fprintf(Spline_file, "\n");
        }
        // Close spline data file
        fclose(Spline_file);
        printf("File created: %s \n", filename);
      }
    }
  }
  return 0;
}
