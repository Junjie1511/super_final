#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <time.h>
#include <algorithm>
#include <functional> 
using namespace std;

class GeometryTools {
public:
    struct Point {
        double x, y;
        Point(double x_ = 0, double y_ = 0) : x(x_), y(y_) {}
    };

    struct Line {
        Point p1, p2;
        Line(Point p1_, Point p2_) : p1(p1_), p2(p2_) {}
    };

    static double distance(const Point& p1, const Point& p2) {
        return sqrt((p2.x - p1.x) * (p2.x - p1.x) + (p2.y - p1.y) * (p2.y - p1.y));
    }

    static bool isInRegion(double x, double y) { 
        // Check if the point is inside the region defined by y^2 < x < 1
        return (y*y < x && x < 1);
    }

    static double computeIntersectionArea(double x, double y, double h1, double h2) {
        double x1 = x - h1/2, x2 = x + h1/2;
        double y1 = y - h2/2, y2 = y + h2/2;
        double cellArea = h1 * h2;

        // Define the boundaries of the parabolic region using function
        function<double(double)> leftBoundary = [](double yVal) { return yVal * yVal; };
        double rightBoundary = 1.0;

        // Calculate intersection points with the parabola and the line x=1
        vector<double> intersectionsY;
        for (double curY : {y1, y2}) {
            if (curY*curY <= rightBoundary && curY*curY >= leftBoundary(curY)) { // 注意这里也使用了 leftBoundary 作为函数调用
                intersectionsY.push_back(curY);
            }
        }

        // Sort the intersection points to ensure we are calculating the correct area
        sort(intersectionsY.begin(), intersectionsY.end());

        double area = 0.0;
        if (!intersectionsY.empty()) {
            // Compute the area between the parabola and the rectangle's edges
            for (size_t i = 0; i < intersectionsY.size() - 1; ++i) {
                double lowerY = intersectionsY[i];
                double upperY = intersectionsY[i + 1];
                double lowerX = max(leftBoundary(lowerY), x1);  
                double upperX = min(rightBoundary, x2);
                area += (upperX - lowerX) * (upperY - lowerY);
            }
        }

        // Add any remaining rectangular area that falls within the bounds
        if (x1 < rightBoundary && x2 > leftBoundary(max(y1, y2))) { // 使用 max
            area += (min(x2, rightBoundary) - max(x1, leftBoundary(min(y1, y2)))) * abs(h2); // 使用 min 和 max
        }

        // Subtract areas outside the parabola but within the rectangle
        for (double curY : {y1, y2}) {
            if (curY*curY < x1 && curY*curY > x2) {
                double segmentHeight = min(abs(curY - y1), abs(curY - y2)); // 使用 abs 和 min
                area -= (leftBoundary(curY) - x1) * segmentHeight;
            }
        }

        return min(area, cellArea); // Ensure we don't exceed the cell's total area
    }

    static double computeIntersectionLength(const Point& p1, const Point& p2) {
        if (abs(p1.x - p2.x) < 1e-10) {
            // Vertical line: calculate intersection with parabola and line x=1
            double x = p1.x;
            if (x >= 1.0 || x <= 0.0) return 0.0; // Outside the region

            double y_min = sqrt(x); // Intersection with parabola
            double y_max = min(sqrt(1.0), max(p1.y, p2.y)); // Intersection with top edge or line x=1

            double y_start = max(-sqrt(x), min(p1.y, p2.y)); // Intersection with bottom edge or parabola
            double y_end = min(y_max, y_min);

            return max(0.0, y_end - y_start);
        }
        if (abs(p1.y - p2.y) < 1e-10) {
            double y = p1.y;
            double ySquared = y * y;

            if (ySquared >= 1.0) return 0.0; // Outside the region

            double x_left = ySquared;
            double x_right = 1.0;

            double x_min = min(p1.x, p2.x);
            double x_max = max(p1.x, p2.x);

            return max(0.0, min(x_max, x_right) - max(x_min, x_left));
        }

        return 0.0; 
    }
};

class PoissonSolver {
private:
    int global_M, global_N;
    double h1, h2;
    double epsilon;
    vector<double> w;    
    vector<double> a, b;
    vector<double> F;    
    vector<double> r;    
    vector<double> Ar;  

    inline int idx(int i, int j) const {
        return i * (global_N + 2) + j;
    }

public:
    PoissonSolver(int global_M_, int global_N_) :
        global_M(global_M_), global_N(global_N_) {

        h1 = 6.0 / global_M_;
        h2 = 4.0 / global_N_;
        epsilon = max(h1, h2) * max(h1, h2);

        int local_size = (global_M_ + 2) * (global_N_ + 2);
        w.resize(local_size, 0.0);
        a.resize(local_size, 0.0);
        b.resize(local_size, 0.0);
        F.resize(local_size, 0.0);
        r.resize(local_size, 0.0);
        Ar.resize(local_size, 0.0);

        computeF();
        computeCoefficients();
    }

    void computeF() {
        #pragma omp parallel for collapse(2)
        for (int i = 1; i <= global_M; i++) {
            for (int j = 1; j <= global_N; j++) {
                double x = -3 + i * h1;
                double y = 0 + j * h2;
                double S_ij = GeometryTools::isInRegion(x, y) ? GeometryTools::computeIntersectionArea(x, y, h1, h2) : 0.0;
                F[idx(i,j)] = S_ij / (h1 * h2);
            }
        }
    }

    void computeCoefficients() {
        #pragma omp parallel for collapse(2)
        for (int i = 1; i <= global_M+1; i++) { 
            for (int j = 1; j <= global_N+1; j++) {
                double x = -3 + (i - 0.5) * h1;
                GeometryTools::Point p1(x, 0 + (j - 0.5) * h2);
                GeometryTools::Point p2(x, 0 + (j + 0.5) * h2);
                
                double l_ij = GeometryTools::computeIntersectionLength(p1, p2);
                if (abs(l_ij - h2) < 1e-10) {
                    a[idx(i,j)] = 1.0;
                } else if (l_ij < 1e-10) {
                    a[idx(i,j)] = 1.0 / epsilon;
                } else {
                    a[idx(i,j)] = l_ij / h2 + (1.0 - l_ij / h2) / epsilon;
                }

                double y = 0 + (j - 0.5) * h2;
                GeometryTools::Point p3(-3 + (i - 0.5) * h1, y);
                GeometryTools::Point p4(-3 + (i + 0.5) * h1, y);
                
                l_ij = GeometryTools::computeIntersectionLength(p3, p4);
                if (abs(l_ij - h1) < 1e-10) {
                    b[idx(i,j)] = 1.0;
                } else if (l_ij < 1e-10) {
                    b[idx(i,j)] = 1.0 / epsilon;
                } else {
                    b[idx(i,j)] = l_ij / h1 + (1.0 - l_ij / h1) / epsilon;
                }
            }
        }
    }

    void computeResidual() {
        #pragma omp parallel for collapse(2)
        for (int i = 1; i <= global_M; i++) {
            for (int j = 1; j <= global_N; j++) {
                double Aw = -(a[idx(i+1,j)] * (w[idx(i+1,j)] - w[idx(i,j)]) / (h1 * h1) -
                              a[idx(i,j)] * (w[idx(i,j)] - w[idx(i-1,j)]) / (h1 * h1) +
                              b[idx(i,j+1)] * (w[idx(i,j+1)] - w[idx(i,j)]) / (h2 * h2) -
                              b[idx(i,j)] * (w[idx(i,j)] - w[idx(i,j-1)]) / (h2 * h2));
                r[idx(i,j)] = F[idx(i,j)] - Aw;
            }
        }
    }

    double computeArDotR() {
        #pragma omp parallel for collapse(2)
        for (int i = 1; i <= global_M; i++) {
            for (int j = 1; j <= global_N; j++) {
                Ar[idx(i,j)] = -(a[idx(i+1,j)]*(r[idx(i+1,j)] - r[idx(i,j)])/h1 - 
                                 a[idx(i,j)]*(r[idx(i,j)] - r[idx(i-1,j)])/h1)/h1 -
                                (b[idx(i,j+1)]*(r[idx(i,j+1)] - r[idx(i,j)])/h2 -
                                 b[idx(i,j)]*(r[idx(i,j)] - r[idx(i,j-1)])/h2)/h2;
            }
        }

        double sum = 0.0;
        for (int i = 1; i <= global_M; i++) {
            for (int j = 1; j <= global_N; j++) {
                sum += Ar[idx(i,j)] * r[idx(i,j)];
            }
        }

        return sum * h1 * h2;
    }

    double computeRDotR() {
        double sum = 0.0;
        #pragma omp parallel for collapse(2)
        for (int i = 1; i <= global_M; i++) {
            for (int j = 1; j <= global_N; j++) {
                sum += r[idx(i,j)] * r[idx(i,j)];
            }
        }

        return sum * h1 * h2;
    }

    void solve(double tol = 1e-6) {
        double error = 1.0;
        auto start = chrono::high_resolution_clock::now();
        int iteration_count = 0;
        while (error > tol) {
            computeResidual();
            iteration_count++;

            double rdotr = computeRDotR();
            double ardotr = computeArDotR();

            double tau = rdotr / ardotr;

            double max_dw = 0.0;
            for (int i = 1; i <= global_M; i++) {
                for (int j = 1; j <= global_N; j++) {
                    double dw = tau * r[idx(i,j)];
                    w[idx(i,j)] += dw;
                    max_dw = max(max_dw, abs(dw));
                }
            }
            error = max_dw;
        }

        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> duration = end - start;
        cout << "Iterations: " << iteration_count << endl;
        cout << "Computation time: " << duration.count() << " seconds" << endl;
    }

    void saveToFile(const string& filename) {
        ofstream out(filename);
        out << scientific << setprecision(10);

        for (int i = 1; i <= global_M; i++) {
            double x = -3 + i * h1;
            for (int j = 1; j <= global_N; j++) {
                double y = 0 + j * h2;
                out << x << " " << y << " " << w[idx(i,j)] << endl;
            }
            out << endl;
        }
        out.close();
    }
};

int main() {
    vector<pair<int, int>> grids = {{10, 10},  {80, 90}, {160, 180}};

    for (const auto& grid : grids) {
        cout << "\nSolving for grid " << grid.first << "x" << grid.second << endl;

        PoissonSolver solver(grid.first, grid.second);
        solver.solve();

        string filename = "solution_" + to_string(grid.first) + "x" +
                          to_string(grid.second) + ".dat";
        solver.saveToFile(filename);
    }

    return 0;
}