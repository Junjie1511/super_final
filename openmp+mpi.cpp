#include <iostream>
#include <cmath>
#include <vector>
#include <iomanip>
#include <fstream>
#include <time.h>
#include <chrono> 
#include <mpi.h>
#include <algorithm>

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

        // Define the boundaries of the parabolic region
        double leftBoundary = [y](double yVal) { return yVal * yVal; };
        double rightBoundary = 1.0;

        // Calculate intersection points with the parabola and the line x=1
        vector<double> intersectionsY;
        for (double curY : {y1, y2}) {
            if (curY*curY <= rightBoundary && curY*curY >= leftBoundary) {
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
        if (x1 < rightBoundary && x2 > leftBoundary(max(y1, y2))) {
            area += (min(x2, rightBoundary) - max(x1, leftBoundary(min(y1, y2)))) * abs(h2);
        }

        // Subtract areas outside the parabola but within the rectangle
        for (double curY : {y1, y2}) {
            if (curY*curY < x1 && curY*curY > x2) {
                double segmentHeight = min(abs(curY - y1), abs(curY - y2));
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
    int local_M, local_N;   
    int M_start, N_start;    
    double h1, h2;
    double epsilon;
    vector<double> w;   
    vector<double> a, b;
    vector<double> F;    
    vector<double> r;    
    vector<double> Ar;  

    int rank, size;
    MPI_Comm cart_comm;
    int dims[2], periods[2], coords[2];
    int upper, down, right, left; 

    MPI_Datatype column_type;
    MPI_Datatype row_type;   

    inline int idx(int i, int j) const {
        return i * (local_N + 2) + j;
    }

    void createMPITypes() {
        MPI_Type_vector(local_M, 1, local_N + 2, MPI_DOUBLE, &column_type);
        MPI_Type_commit(&column_type);
        MPI_Type_contiguous(local_N, MPI_DOUBLE, &row_type);
        MPI_Type_commit(&row_type);
    }

    void exchangeBoundaries(double* var) {
        MPI_Request requests[8];
        int req_count = 0; 

        vector<double> send_left(local_M), recv_left(local_M);
        vector<double> send_right(local_M), recv_right(local_M);

        for (int i = 0; i < local_M; i++) {
            send_left[i] = var[idx(i + 1, 1)];          
            send_right[i] = var[idx(i + 1, local_N)];    
        }

        if (upper != MPI_PROC_NULL) {
            MPI_Isend(&var[idx(1, 1)], 1, row_type, upper, 0, cart_comm, &requests[req_count++]);
            MPI_Irecv(&var[idx(0, 1)], 1, row_type, upper, 1, cart_comm, &requests[req_count++]);
        }
        if (down != MPI_PROC_NULL) {
            MPI_Isend(&var[idx(local_M, 1)], 1, row_type, down, 1, cart_comm, &requests[req_count++]);
            MPI_Irecv(&var[idx(local_M + 1, 1)], 1, row_type, down, 0, cart_comm, &requests[req_count++]);
        }

        if (left != MPI_PROC_NULL) {
            MPI_Isend(send_left.data(), local_M, MPI_DOUBLE, left, 2, cart_comm, &requests[req_count++]);
            MPI_Irecv(recv_left.data(), local_M, MPI_DOUBLE, left, 3, cart_comm, &requests[req_count++]);
        }
        if (right != MPI_PROC_NULL) {
            MPI_Isend(send_right.data(), local_M, MPI_DOUBLE, right, 3, cart_comm, &requests[req_count++]);
            MPI_Irecv(recv_right.data(), local_M, MPI_DOUBLE, right, 2, cart_comm, &requests[req_count++]);
        }

        if (req_count > 0) {
            MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
        }

        for (int i = 0; i < local_M; i++) {
            if (left != MPI_PROC_NULL) {
                var[idx(i + 1, 0)] = recv_left[i]; 
            }
            if (right != MPI_PROC_NULL) {
                var[idx(i + 1, local_N + 1)] = recv_right[i];
            }
        }
    }

    void computeF() {
        #pragma omp parallel for collapse(2)
        for(int i = 1; i <= local_M; i++) {
            for(int j = 1; j <= local_N; j++) {
                int global_i = M_start + i;
                int global_j = N_start + j;
                if(global_i >= 1 && global_i < global_M && global_j >= 1 && global_j < global_N){
                    double x = -3 + global_i * h1;
                    double y = 0 + global_j * h2;
                    double S_ij = GeometryTools::computeIntersectionArea(x, y, h1, h2); // 确保使用了针对抛物线区域的版本
                    F[idx(i,j)] = S_ij / (h1 * h2);
                } else {
                    F[idx(i,j)] = 0.0; // 处理网格单元不在求解区域内的情况
                }
            }
        }
    }

    void computeCoefficients() {
        #pragma omp parallel for collapse(2)
        for(int i = 1; i <= local_M+1; i++) { 
            for(int j = 1; j <= local_N+1; j++) {
                int global_i = M_start + i;
                int global_j = N_start + j;
                if(global_i >= 1 && global_i <= global_M && global_j >= 1 && global_j <= global_N){
                    double x = -3 + (global_i - 0.5) * h1;
                    GeometryTools::Point p1(x, 0 + (global_j - 0.5) * h2);
                    GeometryTools::Point p2(x, 0 + (global_j + 0.5) * h2);
                    
                    double l_ij = GeometryTools::computeIntersectionLength(p1, p2); // 确保使用了针对抛物线区域的版本
                    if(abs(l_ij - h2) < 1e-10) {
                        a[idx(i,j)] = 1.0;
                    }
                    else if(l_ij < 1e-10) {
                        a[idx(i,j)] = 1.0 / epsilon;
                    }
                    else {
                        a[idx(i,j)] = l_ij / h2 + (1.0 - l_ij / h2) / epsilon;
                    }

                    double y = 0 + (global_j - 0.5) * h2;
                    GeometryTools::Point p3(-3 + (global_i - 0.5) * h1, y);
                    GeometryTools::Point p4(-3 + (global_i + 0.5) * h1, y);
                    
                    l_ij = GeometryTools::computeIntersectionLength(p3, p4); // 确保使用了针对抛物线区域的版本
                    if(abs(l_ij - h1) < 1e-10) {
                        b[idx(i,j)] = 1.0;
                    }
                    else if(l_ij < 1e-10) {
                        b[idx(i,j)] = 1.0 / epsilon;
                    }
                    else {
                        b[idx(i,j)] = l_ij / h1 + (1.0 - l_ij / h1) / epsilon;
                    }
                } else {
                    a[idx(i,j)] = 1.0;
                    b[idx(i,j)] = 1.0;
                }
            }
        }
    }


public:
    PoissonSolver(int global_M_, int global_N_, MPI_Comm cart_comm_) :
        global_M(global_M_), global_N(global_N_), cart_comm(cart_comm_) {

        MPI_Comm_rank(cart_comm, &rank);
        MPI_Comm_size(cart_comm, &size);

        MPI_Cart_get(cart_comm, 2, dims, periods, coords);
        MPI_Cart_shift(cart_comm, 0, 1, &upper, &down);
        MPI_Cart_shift(cart_comm, 1, 1, &left, &right);

        local_M = global_M / dims[0] + (coords[0] < global_M % dims[0] ? 1 : 0);
        local_N = global_N / dims[1] + (coords[1] < global_N % dims[1] ? 1 : 0);

        M_start = coords[0] * (global_M / dims[0]) + std::min(coords[0], global_M % dims[0]);
        N_start = coords[1] * (global_N / dims[1]) + std::min(coords[1], global_N % dims[1]);

        h1 = 6.0 / global_M;
        h2 = 4.0 / global_N;
        epsilon = std::max(h1, h2) * std::max(h1, h2);

        int local_size = (local_M + 2) * (local_N + 2);
        w.resize(local_size, 0.0);
        a.resize(local_size, 0.0);
        b.resize(local_size, 0.0);
        F.resize(local_size, 0.0);
        r.resize(local_size, 0.0);
        Ar.resize(local_size, 0.0);

        createMPITypes();
        computeF();
        computeCoefficients();
    }

    ~PoissonSolver() {
        MPI_Type_free(&column_type);
        MPI_Type_free(&row_type);
    }

    void computeResidual() {
        exchangeBoundaries(w.data());

        for (int i = 1; i <= local_M; i++) {
            for (int j = 1; j <= local_N; j++) {
                int global_i = M_start + i;
                int global_j = N_start + j;

                if (global_i >= 1 && global_i < global_M && global_j >= 1 && global_j < global_N) {
                    double Aw = -(a[idx(i+1,j)] * (w[idx(i+1,j)] - w[idx(i,j)]) / (h1 * h1) -
                                  a[idx(i,j)] * (w[idx(i,j)] - w[idx(i-1,j)]) / (h1 * h1) +
                                  b[idx(i,j+1)] * (w[idx(i,j+1)] - w[idx(i,j)]) / (h2 * h2) -
                                  b[idx(i,j)] * (w[idx(i,j)] - w[idx(i,j-1)]) / (h2 * h2));
                    r[idx(i,j)] = F[idx(i,j)] - Aw;
                } else {
                    r[idx(i,j)] = 0.0;
                }
            }
        }
    }

    double computeArDotR() {
        exchangeBoundaries(r.data());

        #pragma omp parallel for collapse(2) reduction(+:local_sum)
        for (int i = 1; i <= local_M; i++) {
            for (int j = 1; j <= local_N; j++) {
                int global_i = M_start + i;
                int global_j = N_start + j;

                if (global_i >= 1 && global_i < global_M && global_j >= 1 && global_j < global_N) {
                    Ar[idx(i,j)] = -(a[idx(i+1,j)]*(r[idx(i+1,j)] - r[idx(i,j)])/h1 - 
                                     a[idx(i,j)]*(r[idx(i,j)] - r[idx(i-1,j)])/h1)/h1 -
                                    (b[idx(i,j+1)]*(r[idx(i,j+1)] - r[idx(i,j)])/h2 -
                                     b[idx(i,j)]*(r[idx(i,j)] - r[idx(i,j-1)])/h2)/h2;
                } else {
                    Ar[idx(i,j)] = 0.0;
                }
            }
        }

        double local_sum = 0.0;
        for (int i = 1; i <= local_M; i++) {
            for (int j = 1; j <= local_N; j++) {
                int global_i = M_start + i;
                int global_j = N_start + j;

                if (global_i >= 1 && global_i < global_M && global_j >= 1 && global_j < global_N) {
                    local_sum += Ar[idx(i,j)] * r[idx(i,j)];
                }
            }
        }

        return local_sum * h1 * h2;
    }

    double computeRDotR() {
        double local_sum = 0.0;
        for (int i = 1; i <= local_M; i++) {
            for (int j = 1; j <= local_N; j++) {
                int global_i = M_start + i;
                int global_j = N_start + j;

                if (global_i >= 1 && global_i < global_M && global_j >= 1 && global_j < global_N) {
                    local_sum += r[idx(i,j)] * r[idx(i,j)];
                }
            }
        }

        return local_sum * h1 * h2;
    }

    void solve(double tol = 1e-6) {
        double error = 1.0;
        auto start = std::chrono::high_resolution_clock::now();
        int iteration_count = 0;
        while (error > tol) {
            computeResidual();
            iteration_count++;

            double local_results[2] = {computeRDotR(), computeArDotR()};
            double global_results[2];
            MPI_Allreduce(local_results, global_results, 2, MPI_DOUBLE, MPI_SUM, cart_comm);

            double global_rdotr = global_results[0];
            double global_ardotr = global_results[1];

            double tau = global_rdotr / global_ardotr;

            double local_error = 0.0;
            for (int i = 1; i <= local_M; i++) {
                for (int j = 1; j <= local_N; j++) {
                    double dw = tau * r[idx(i,j)];
                    w[idx(i,j)] += dw;
                    local_error = std::max(local_error, std::abs(dw));
                }
            }
            MPI_Allreduce(&local_error, &error, 1, MPI_DOUBLE, MPI_MAX, cart_comm);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        if (rank == 0) {
            cout << "Iterations: " << iteration_count << std::endl;
            cout << "Computation time: " << duration.count() << " seconds" << std::endl;
        }
    }

    void saveToFile(const std::string& filename) {
        vector<int> sendcounts(size);
        vector<int> displs(size);
        vector<double> global_w;

        int local_data_size = local_M * local_N;

        // 收集所有进程的数据大小
        MPI_Gather(&local_data_size, 1, MPI_INT, 
                   sendcounts.data(), 1, MPI_INT, 0, cart_comm);

        if (rank == 0) {
            displs[0] = 0;
            for (int i = 1; i < size; i++) {
                displs[i] = displs[i-1] + sendcounts[i-1];
            }
            global_w.resize(displs[size-1] + sendcounts[size-1]);
        }

        // 准备本地数据
        vector<double> local_data(local_M * local_N);
        #pragma omp parallel for
        for (int i = 0; i < local_M; i++) {
            for (int j = 0; j < local_N; j++) {
                local_data[i*local_N + j] = w[idx(i+1,j+1)];
            }
        }

        // 收集所有数据
        MPI_Gatherv(local_data.data(), local_data_size, MPI_DOUBLE,
                    global_w.data(), sendcounts.data(), displs.data(), 
                    MPI_DOUBLE, 0, cart_comm);

        if (rank == 0) {
            ofstream out(filename);
            out << std::scientific << std::setprecision(10);

            int index = 0;
            for (int p = 0; p < size; p++) {
                int proc_coords[2];
                MPI_Cart_coords(cart_comm, p, 2, proc_coords);
                int proc_M = (global_M + dims[0] - 1) / dims[0];
                int proc_N = (global_N + dims[1] - 1) / dims[1];

                if (proc_coords[0] == dims[0] - 1) {
                    proc_M = global_M - (dims[0] - 1) * proc_M;
                }
                if (proc_coords[1] == dims[1] - 1) {
                    proc_N = global_N - (dims[1] - 1) * proc_N;
                }

                int M_offset = proc_coords[0] * ((global_M + dims[0] - 1) / dims[0]);
                int N_offset = proc_coords[1] * ((global_N + dims[1] - 1) / dims[1]);

                for (int i = 0; i < proc_M; i++) {
                    double x = -3 + (M_offset + i) * h1;
                    for (int j = 0; j < proc_N; j++) {
                        double y = 0 + (N_offset + j) * h2;
                        out << x << " " << y << " " << global_w[displs[p] + i*proc_N + j] << endl;
                    }
                    out << endl;
                }
            }
            out.close();
        }
    }
};


int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);

    int periods[2] = {0, 0}; 
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

    vector<pair<int,int>> grids = {{10,10},{39,39},{29,29},{40,40},{160,180}};

    for(const auto& grid : grids) {
        if(rank == 0) {
            cout << "\nSolving for grid " << grid.first << "x" << grid.second << endl;
        }

        PoissonSolver solver(grid.first, grid.second, cart_comm);
        solver.solve();

        string filename = "solution_mpi_" + to_string(grid.first) + "x" +
                          to_string(grid.second) + ".dat";
        solver.saveToFile(filename);
    }
    MPI_Finalize();
    return 0;
}
