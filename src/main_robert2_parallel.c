/*
 * main_robert2_parallel.c
 *
 * This program generates a unipartite network (S1) representing the structure, as well as a bipartite network (S1*S1)
 * that shows the relations between nodes (Type1) and features (Type2).
 * In the bipartite network, nodes of Type1 have the same Thetas as in the unipartite network. Their kappas are correlated
 * with the kappas from the structural network. The number of nodes of Type1 in the bipartite network is equal to the number
 * of non-zero degree nodes in the unipartite network.
 *
 * Usage:
 *   ./main_run_parallel [Beta_s] [gamma_s] [Ns_obs] [kmean_s] [gamma_n] [kmean_n] [gamma_f] [N_f] [Beta_bi] [nu] [alpha] [NC] [outfolder] [t]
 *
 * Compile with OpenMP enabled:
 *   gcc -lm -O3 -fopenmp main_robert2_parallel.c -o main_run_parallel
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <time.h>
 #include <math.h>
 #include <string.h>
 #include <sys/stat.h>
 #include <sys/types.h>
 #include <errno.h>
 #ifdef _OPENMP
 #include <omp.h>
 #endif
 
 // Global pointer for thread-local seeds for rand_r
 #ifdef _OPENMP
 unsigned int *thread_seeds = NULL;
 #endif
 
 // Function prototypes
 double rand_easy(double a, double b);
 double Prob_Connection(double kappa1, double kappa2, double theta1, double theta2, double B, double mu, double Rad);
 double Random_powerlaw(double gamma, double k0);
 void read_file(const char *file_name, int *Nodes, int *Links);
 double get_fun(double r, double k1, double k2, double k1_min, double k2_min, double gamma1, double gamma2, double nu);
 double bisect(double r, double x1, double x2, double err, double k1, double k1_min, double k2_min, double gamma1, double gamma2, double nu);
 double get_fun_truncated(double r, double k1, double k2, double k1_min, double k2_min,
                            double k1_c, double k2_c, double gamma1, double gamma2, double nu);
 double bisect_truncated(double r, double x1, double x2, double err, double k1, double k1_min, double k2_min,
                           double k1_c, double k2_c, double gamma1, double gamma2, double nu);
 void swap(int* xp, int* yp);
 void selectionSort(int arr[], int n);
 double Delta_Theta(double Theta1, double Theta2);
 void Assign_Labels(double *Theta_s, int NC, int *Label, int NZ, float R, float alpha);
 
 double rand_easy(double a, double b)
 {
     double range = b - a;
 #ifdef _OPENMP
     unsigned int seed = thread_seeds[omp_get_thread_num()];
     double r = (((double) rand_r(&seed)) / ((double) RAND_MAX)) * range + a;
     thread_seeds[omp_get_thread_num()] = seed;
     return r;
 #else
     return ((((double)rand() / (double)RAND_MAX) * range) + a);
 #endif
 }
 
 double Prob_Connection(double kappa1, double kappa2, double theta1, double theta2, double B, double mu, double Rad)
 {
     double Delta_T = M_PI - fabs(M_PI - fabs(theta1 - theta2));
     double X = (Rad * Delta_T) / (mu * kappa1 * kappa2);
     double result = 1 / (1 + pow(X, B));
     return result;
 }
 
 double Random_powerlaw(double gamma, double k0)
 {
     double prob = rand_easy(0, 1);
     double value = k0 / (pow((1 - prob), (1 / (gamma - 1))));
     return value;
 }
 
 void read_file(const char *file_name, int *Nodes, int *Links)
 {
     FILE *myfile = fopen(file_name, "r");
     int Num_Nodes = 0;
     int Num_links = 0;
     int Node1, Node2;
     while (1)
     {
         fscanf(myfile, "%d %d", &Node1, &Node2);
         if (feof(myfile))
             break;
         if (Node1 > Num_Nodes)
             Num_Nodes = Node1;
         if (Node2 > Num_Nodes)
             Num_Nodes = Node2;
         Num_links++;
     }
     *Nodes = Num_Nodes;
     *Links = Num_links;
     fclose(myfile);
 }
 
 double get_fun(double r, double k1, double k2, double k1_min, double k2_min, double gamma1, double gamma2, double nu)
 {
     double phi1, phi2, C, res, temp;
     phi1 = -log(1 - pow((k1_min / k1), (gamma1 - 1)));
     phi2 = -log(1 - pow((k2_min / k2), (gamma2 - 1)));
     C = (pow(phi1, (nu / (1 - nu))) * k1_min * pow(k1, gamma1)) /
         ((k1_min * pow(k1, gamma1)) - (pow(k1_min, gamma1) * k1));
     temp = pow(phi1, (1 / (1 - nu))) + pow(phi2, (1 / (1 - nu)));
     res = (exp(-pow(temp, (1 - nu))) * pow(temp, -nu) * C) - r;
     return res;
 }
 
 double bisect(double r, double x1, double x2, double err, double k1, double k1_min, double k2_min, double gamma1, double gamma2, double nu)
 {
     double f1, f2, f, x;
     while (1)
     {
         f1 = get_fun(r, k1, x1, k1_min, k2_min, gamma1, gamma2, nu);
         f2 = get_fun(r, k1, x2, k1_min, k2_min, gamma1, gamma2, nu);
         if (f1 * f2 > 0 || isnan(f1) || isnan(f2))
         {
             printf("\n initial guesses are wrong!\n");
             return nan("");
         }
         x = (x1 + x2) / 2;
         if (((x2 - x1) / x) < err)
         {
             f = get_fun(r, k1, x, k1_min, k2_min, gamma1, gamma2, nu);
             return x;
         }
         f = get_fun(r, k1, x, k1_min, k2_min, gamma1, gamma2, nu);
         if ((f * f1) > 0)
         {
             x1 = x;
             f1 = f;
         }
         else
         {
             x2 = x;
             f2 = f;
         }
     }
 }
 
 double get_fun_truncated(double r, double k1, double k2, double k1_min, double k2_min,
                            double k1_c, double k2_c, double gamma1, double gamma2, double nu)
 {
     double phi1, phi2, C, res, temp;
     phi1 = -log(1 - ((pow(k1, (1 - gamma1)) - pow(k1_c, (1 - gamma1))) /
                      (pow(k1_min, (1 - gamma1)) - pow(k1_c, (1 - gamma1)))));
     phi2 = -log(1 - ((pow(k2, (1 - gamma2)) - pow(k2_c, (1 - gamma2))) /
                      (pow(k2_min, (1 - gamma2)) - pow(k2_c, (1 - gamma2)))));
     C = pow(phi1, (nu / (1 - nu))) *
         (1 / (1 - ((pow(k1, (1 - gamma1)) - pow(k1_c, (1 - gamma1))) /
                   (pow(k1_min, (1 - gamma1)) - pow(k1_c, (1 - gamma1))))));
     temp = pow(phi1, (1 / (1 - nu))) + pow(phi2, (1 / (1 - nu)));
     res = (exp(-pow(temp, (1 - nu))) * pow(temp, -nu) * C) - r;
     return res;
 }
 
 double bisect_truncated(double r, double x1, double x2, double err, double k1, double k1_min, double k2_min,
                           double k1_c, double k2_c, double gamma1, double gamma2, double nu)
 {
     double f1, f2, f, x;
     f1 = get_fun_truncated(r, k1, x1, k1_min, k2_min, k1_c, k2_c, gamma1, gamma2, nu);
     f2 = get_fun_truncated(r, k1, x2, k1_min, k2_min, k1_c, k2_c, gamma1, gamma2, nu);
     if (f1 * f2 > 0)
     {
         printf("%f \t %f \n", f1, f2);
         while (1)
         {
             x1 = rand_easy(k2_min, k2_c);
             x2 = rand_easy(k2_min, k2_c);
             f1 = get_fun_truncated(r, k1, x1, k1_min, k2_min, k1_c, k2_c, gamma1, gamma2, nu);
             f2 = get_fun_truncated(r, k1, x2, k1_min, k2_min, k1_c, k2_c, gamma1, gamma2, nu);
             if (f1 * f2 < 0)
             {
                 printf("******Yes!!********\n");
                 printf("%f \t %f \n", f1, f2);
                 break;
             }
         }
     }
     while (1)
     {
         f1 = get_fun_truncated(r, k1, x1, k1_min, k2_min, k1_c, k2_c, gamma1, gamma2, nu);
         f2 = get_fun_truncated(r, k1, x2, k1_min, k2_min, k1_c, k2_c, gamma1, gamma2, nu);
         x = (x1 + x2) / 2;
         if (((x2 - x1) / x) < err)
         {
             f = get_fun_truncated(r, k1, x, k1_min, k2_min, k1_c, k2_c, gamma1, gamma2, nu);
             return x;
         }
         f = get_fun_truncated(r, k1, x, k1_min, k2_min, k1_c, k2_c, gamma1, gamma2, nu);
         if ((f * f1) > 0)
         {
             x1 = x;
             f1 = f;
         }
         else
         {
             x2 = x;
             f2 = f;
         }
     }
 }
 
 void swap(int* xp, int* yp)
 {
     int temp = *xp;
     *xp = *yp;
     *yp = temp;
 }
 
 void selectionSort(int arr[], int n)
 {
     int i, j, min_idx;
     for (i = 0; i < n - 1; i++) {
         min_idx = i;
         for (j = i + 1; j < n; j++)
             if (arr[j] < arr[min_idx])
                 min_idx = j;
         swap(&arr[min_idx], &arr[i]);
     }
 }
 
 double Delta_Theta(double Theta1, double Theta2)
 {
     return (M_PI - fabs(M_PI - fabs(Theta1 - Theta2)));
 }
 
 void Assign_Labels(double *Theta_s, int NC, int *Label, int NZ, float R, float alpha)
 {
     float c_centers[NC];
     double e = 2 * M_PI;
     float r;
 
     alpha = -1 * alpha;
 
     printf("\n");
     for (int i = 0; i < NC; i++) {
         c_centers[i] = rand_easy(0, e);
         printf("Center number %d is located at: %f\n", i, c_centers[i]);
     }
 
     #pragma omp parallel for
     for (int node = 0; node < NZ; node++) {
         float d[NC];
         float pr[NC];
         float delta_Th;
         float d_sum = 0;
         for (int l = 0; l < NC; l++) {
             delta_Th = Delta_Theta(Theta_s[node], c_centers[l]);
             d[l] = R * delta_Th;
             d_sum += pow(d[l], alpha);
         }
         for (int l = 0; l < NC; l++) {
             if(l == 0)
                 pr[l] = pow(d[l], alpha) / d_sum;
             else
                 pr[l] = (pow(d[l], alpha) / d_sum) + pr[l-1];
         }
         r = rand_easy(0, 1);
         for (int l = 0; l < NC; l++) {
             if ((l == 0 && r >= 0 && r < pr[l]) ||
                 (l > 0 && r >= pr[l-1] && r < pr[l])) {
                 Label[node] = l;
                 break;
             }
         }
     }
 }
 
 int main(int argc, char *argv[])
 {
     /**
      * Usage:
      * ./main_run_parallel [Beta_s] [gamma_s] [Ns_obs] [kmean_s] [gamma_n] [kmean_n] [gamma_f] [N_f] [Beta_bi] [nu] [alpha] [NC] [outfolder] [t]
      */
     if (argc != 15) {
         printf("Please provide all 14 parameters\n");
         exit(0);
     }
 
     // S1 model parameters
     float Beta_s = atof(argv[1]);
     float gamma_s = atof(argv[2]);
     int Ns_obs = atoi(argv[3]);
     int Ns;
     float kmean_s = atof(argv[4]);
     float Corr_k0_s = (1 - (1 / (float)Ns_obs)) / (1 - pow(Ns_obs, ((2 - gamma_s) / (gamma_s - 1))));
     float k0_s = (((gamma_s - 2) * kmean_s) / (gamma_s - 1)) * Corr_k0_s;
     float kc_s = k0_s * (pow(Ns_obs, (1 / (gamma_s - 1))));
     int delta_s = 1;
     float R_s = Ns_obs / (2 * M_PI * delta_s);
     float mu_s = (Beta_s * sin(M_PI / Beta_s)) / (2 * delta_s * kmean_s * M_PI);
     float alpha_s = atof(argv[11]);
     int NC = atoi(argv[12]);
 
     // S1*S1 model parameters for nodes of Type 1 and Type 2
     float gamma_n = atof(argv[5]);
     int N_n, N_n_obs;
     float kmean_n = atof(argv[6]);
     float Corr_k0_n, k0_n, kc_n;
     int delta_n = 1;
 
     float gamma_f = atof(argv[7]);
     int N_f, N_f_obs = atoi(argv[8]);
     float kmean_f, Corr_k0_f, k0_f, kc_f;
 
     float Beta_bi = atof(argv[9]);
     float R_bi, mu_bi;
 
     float Err = 0.001;
     float nu = atof(argv[10]);
 
     // New parameter: number of threads
     int t = atoi(argv[14]);
 #ifdef _OPENMP
     thread_seeds = (unsigned int*) malloc(t * sizeof(unsigned int));
     for (int i = 0; i < t; i++) {
         thread_seeds[i] = (unsigned int) time(NULL) + i;
     }
 #endif
     omp_set_num_threads(t);
 
     // Create output folder and filenames
     FILE *fp, *fp2, *fp3, *fp4;
     char Node_s[33], B_s[33], g_s[33], K_s[33], a_s[33], num_c[33];
     sprintf(Node_s, "%d", Ns_obs);
     sprintf(B_s, "%2.2f", Beta_s);
     sprintf(g_s, "%2.2f", gamma_s);
     sprintf(K_s, "%2.2f", kmean_s);
     sprintf(a_s, "%2.2f", alpha_s);
     sprintf(num_c, "%d", NC);
 
     char *foldername = argv[13];
     if (mkdir(foldername, S_IRWXU | S_IRWXG | S_IRWXO) == -1) {
         printf("Error: %s\n", strerror(errno));
     }
 
     char S1_Net_Add[500] = "";
     strcat(S1_Net_Add, foldername);
     strcat(S1_Net_Add, "/Net_N_");
     strcat(S1_Net_Add, Node_s);
     strcat(S1_Net_Add, "_g_");
     strcat(S1_Net_Add, g_s);
     strcat(S1_Net_Add, "_B_");
     strcat(S1_Net_Add, B_s);
     strcat(S1_Net_Add, "_k_");
     strcat(S1_Net_Add, K_s);
     strcat(S1_Net_Add, "_a_");
     strcat(S1_Net_Add, a_s);
     strcat(S1_Net_Add, "_nc_");
     strcat(S1_Net_Add, num_c);
     strcat(S1_Net_Add, ".unipartite.edgelist");
 
     char Node_1[33], Node_2[33], B_bi[33], g1_bi[33], g2_bi[33], k1_bi[33], nu_bi[33];
     sprintf(Node_1, "%d", Ns_obs);
     sprintf(Node_2, "%d", N_f_obs);
     sprintf(B_bi, "%2.2f", Beta_bi);
     sprintf(g1_bi, "%2.2f", gamma_n);
     sprintf(g2_bi, "%2.2f", gamma_f);
     sprintf(k1_bi, "%2.2f", kmean_n);
     sprintf(nu_bi, "%2.2f", nu);
 
     char Bi_Net_Add[500] = "";
     strcat(Bi_Net_Add, foldername);
     strcat(Bi_Net_Add, "/Net_N_");
     strcat(Bi_Net_Add, Node_1);
     strcat(Bi_Net_Add, "_N2_");
     strcat(Bi_Net_Add, Node_2);
     strcat(Bi_Net_Add, "_g1_");
     strcat(Bi_Net_Add, g1_bi);
     strcat(Bi_Net_Add, "_g2_");
     strcat(Bi_Net_Add, g2_bi);
     strcat(Bi_Net_Add, "_B_");
     strcat(Bi_Net_Add, B_bi);
     strcat(Bi_Net_Add, "_k1_");
     strcat(Bi_Net_Add, k1_bi);
     strcat(Bi_Net_Add, "_nu_");
     strcat(Bi_Net_Add, nu_bi);
     strcat(Bi_Net_Add, ".bipartite.edgelist");
 
     char U_Coordinate[500] = "";
     strcat(U_Coordinate, foldername);
     strcat(U_Coordinate, "/Net_N_");
     strcat(U_Coordinate, Node_s);
     strcat(U_Coordinate, "_g_");
     strcat(U_Coordinate, g_s);
     strcat(U_Coordinate, "_B_");
     strcat(U_Coordinate, B_s);
     strcat(U_Coordinate, "_k_");
     strcat(U_Coordinate, K_s);
     strcat(U_Coordinate, "_a_");
     strcat(U_Coordinate, a_s);
     strcat(U_Coordinate, "_nc_");
     strcat(U_Coordinate, num_c);
     strcat(U_Coordinate, ".unipartite.coordinates");
 
     char Bi_Coordinate[500] = "";
     strcat(Bi_Coordinate, foldername);
     strcat(Bi_Coordinate, "/Net_N_");
     strcat(Bi_Coordinate, Node_1);
     strcat(Bi_Coordinate, "_N2_");
     strcat(Bi_Coordinate, Node_2);
     strcat(Bi_Coordinate, "_g1_");
     strcat(Bi_Coordinate, g1_bi);
     strcat(Bi_Coordinate, "_g2_");
     strcat(Bi_Coordinate, g2_bi);
     strcat(Bi_Coordinate, "_B_");
     strcat(Bi_Coordinate, B_bi);
     strcat(Bi_Coordinate, "_k1_");
     strcat(Bi_Coordinate, k1_bi);
     strcat(Bi_Coordinate, "_nu_");
     strcat(Bi_Coordinate, nu_bi);
     strcat(Bi_Coordinate, ".bipartite.coordinates");
 
     /*~~~~~~~~~~~~~~~~~~~~~~~~ S1 Model ~~~~~~~~~~~~~~~~~~~~~~~~*/
     Ns = Ns_obs;
     double *kappa_s = (double *) malloc(Ns * sizeof(double));
     #pragma omp parallel for
     for (int i = 0; i < Ns; i++) {
         double kappa;
         while (1) {
             kappa = Random_powerlaw(gamma_s, k0_s);
             if (kappa <= kc_s && kappa >= k0_s) {
                 kappa_s[i] = kappa;
                 break;
             }
         }
     }
     double *Theta_s = (double *) malloc(Ns * sizeof(double));
     #pragma omp parallel for
     for (int i = 0; i < Ns; i++) {
         double e = 2 * M_PI;
         Theta_s[i] = rand_easy(0, e);
     }
 
     int iteration = 0;
     printf("Playing with Mu to get a unipartite network with a desired average degree....!\n");
     printf("=====================================\n");
     double AVG_DEG; // declare AVG_DEG here
 
     while (1)
     {
         printf("\n iteration # %d", iteration);
         printf("\n Mu is : %f", mu_s);
         fp = fopen(S1_Net_Add, "w+");
         if (fp == NULL) {
             perror("Failed: ");
             return 1;
         }
         double Num_links = 0;
         #pragma omp parallel for reduction(+:Num_links) schedule(dynamic)
         for (int ii = 0; ii < (Ns - 1); ii++) {
             for (int jj = ii + 1; jj < Ns; jj++) {
                 double prob = Prob_Connection(kappa_s[ii], kappa_s[jj], Theta_s[ii], Theta_s[jj], Beta_s, mu_s, R_s);
                 double r = rand_easy(0, 1);
                 if (r <= prob) {
                     #pragma omp critical
                     {
                         fprintf(fp, "%d \t %d\n", ii, jj);
                     }
                     Num_links += 1;
                 }
             }
         }
         fclose(fp);
         AVG_DEG = 2 * Num_links / (double) Ns_obs;
         printf("\n The average degree of nodes is: %f \n", AVG_DEG);
         printf("~~~~~~~~~~~~~");
         if (fabs(AVG_DEG - kmean_s) < 0.1)
             break;
         if (AVG_DEG < kmean_s)
             mu_s = mu_s + (0.01 * mu_s);
         if (AVG_DEG > kmean_s)
             mu_s = mu_s - (0.005 * mu_s);
         iteration++;
     }
 
     fp3 = fopen(S1_Net_Add, "r");
     if (fp3 == NULL) {
         perror("Failed: ");
         return 1;
     }
     int NGZ = 0, Ver_num, Edge_num;
     read_file(S1_Net_Add, &Ver_num, &Edge_num);
     int **Edgelist_s = (int **) malloc(Edge_num * sizeof(int *));
     for (int i = 0; i < Edge_num; i++) {
         Edgelist_s[i] = (int *) malloc(2 * sizeof(int));
     }
     int ind = 0;
     while (1)
     {
         if (feof(fp3))
             break;
         fscanf(fp3, "%d %d", &Edgelist_s[ind][0], &Edgelist_s[ind][1]);
         ind++;
     }
     fclose(fp3);
     printf("\n==============The properties of the generated unipartite networks==============\n");
     for (int ii = 0; ii <= Ver_num; ii++) {
         int found = 0;
         for (int jj = 0; jj < Edge_num; jj++) {
             if (Edgelist_s[jj][0] == ii || Edgelist_s[jj][1] == ii) {
                 found = 1;
                 break;
             }
         }
         if (found == 1)
             NGZ++;
     }
     printf("The number of nodes with degree greater than zero is: %d\n", NGZ);
     printf("The number of Edges is: %d\n", Edge_num);
     double NGZ0 = NGZ;
     AVG_DEG = 2 * Edge_num / NGZ0;
     printf("The average degree of nodes is: %f", AVG_DEG);
 
     int *Nodelist_s = (int *) malloc(NGZ * sizeof(int));
     int *Nodelabel_s = (int *) malloc(NGZ * sizeof(int));
     for (int i = 0; i < NGZ; i++) {
         Nodelabel_s[i] = i;
         Nodelist_s[i] = -1;
     }
     int iindd = 0;
     for (int i = 0; i < Edge_num; i++) {
         int found = 0;
         for (int j = 0; j < NGZ; j++) {
             if (Nodelist_s[j] == Edgelist_s[i][0]) {
                 found = 1;
                 break;
             }
         }
         if (found == 0) {
             Nodelist_s[iindd] = Edgelist_s[i][0];
             iindd++;
         }
     }
     for (int i = 0; i < Edge_num; i++) {
         int found = 0;
         for (int j = 0; j < NGZ; j++) {
             if (Nodelist_s[j] == Edgelist_s[i][1]) {
                 found = 1;
                 break;
             }
         }
         if (found == 0) {
             Nodelist_s[iindd] = Edgelist_s[i][1];
             iindd++;
         }
     }
     selectionSort(Nodelist_s, NGZ);
 
     fp2 = fopen(S1_Net_Add, "w+");
     if (fp2 == NULL) {
         perror("Failed: ");
         return 1;
     }
     int temp0, temp1;
     for (int ii = 0; ii < Edge_num; ii++) {
         temp0 = 0;
         temp1 = 0;
         for (int jj = 0; jj < NGZ; jj++) {
             if (Nodelist_s[jj] == Edgelist_s[ii][0]) {
                 temp0 = jj;
                 break;
             }
         }
         for (int kk = 0; kk < NGZ; kk++) {
             if (Nodelist_s[kk] == Edgelist_s[ii][1]) {
                 temp1 = kk;
                 break;
             }
         }
         fprintf(fp2, "%d \t %d\n", Nodelabel_s[temp0], Nodelabel_s[temp1]);
     }
     fclose(fp2);
 
     double *New_kappa_s = (double *) malloc(NGZ * sizeof(double));
     double *New_Theta_s = (double *) malloc(NGZ * sizeof(double));
     int *Label = (int *) malloc(NGZ * sizeof(int));
     for (int bb = 0; bb < NGZ; bb++) {
         New_kappa_s[bb] = kappa_s[Nodelist_s[bb]];
         New_Theta_s[bb] = Theta_s[Nodelist_s[bb]];
     }
     Assign_Labels(New_Theta_s, NC, Label, NGZ, R_s, alpha_s);
     fp2 = fopen(U_Coordinate, "w+");
     if (fp2 == NULL) {
         perror("Failed: ");
         return 1;
     }
     for (int bb = 0; bb < NGZ; bb++) {
         fprintf(fp2, "%d\t%f\t%f\t%d\n", bb, New_kappa_s[bb], New_Theta_s[bb], Label[bb]);
     }
     fclose(fp2);
 
     free(kappa_s);
     free(Theta_s);
     for (int i = 0; i < Edge_num; i++) {
         free(Edgelist_s[i]);
     }
     free(Edgelist_s);
     free(Nodelabel_s);
     free(Nodelist_s);
     free(Label);
 
     /*~~~~~~~~~~~~~~~~~~~~ S1*S1 Model (Bipartite Network) ~~~~~~~~~~~~~~~~~~~~~*/
     printf("\n=========================Bipartite network=========================\n");
     N_n = NGZ;
     N_n_obs = NGZ;
     Corr_k0_n = (1 - (1 / (float) N_n_obs)) / (1 - pow(N_n_obs, ((2 - gamma_n) / (gamma_n - 1))));
     k0_n = (((gamma_n - 2) * kmean_n) / (gamma_n - 1)) * Corr_k0_n;
     kc_n = k0_n * (pow(N_n_obs, (1 / (gamma_n - 1))));
     double *kappa_n = (double *) malloc(N_n_obs * sizeof(double));
 
     double st_p = k0_n + 0.000005;
     double end_p = kc_n;
 
     #pragma omp parallel for
     for (int ii = 0; ii < N_n_obs; ii++) {
         double localRnd = rand_easy(0, 1);
         double temp = bisect_truncated(localRnd, st_p, end_p, Err, New_kappa_s[ii], k0_s, k0_n, kc_s, kc_n, gamma_s, gamma_n, nu);
         kappa_n[ii] = temp;
     }
 
     double *Theta_n = (double *) malloc(N_n * sizeof(double));
     for (int i = 0; i < N_n; i++) {
         Theta_n[i] = New_Theta_s[i];
     }
 
     kmean_f = (kmean_n * N_n) / N_f_obs;
     Corr_k0_f = (1 - (1 / (float) N_f_obs)) / (1 - pow(N_f_obs, ((2 - gamma_f) / (gamma_f - 1))));
     k0_f = (((gamma_f - 2) * kmean_f) / (gamma_f - 1)) * Corr_k0_f;
     kc_f = k0_f * (pow(N_f_obs, (1 / (gamma_f - 1))));
     N_f = N_f_obs;
 
     printf("\nThe number of nodes of Type2 is: %d \n", N_f);
     printf("Their desired average degree of nodes of Type2 is: %f\n", kmean_f);
 
     double *kappa_f = (double *) malloc(N_f * sizeof(double));
     #pragma omp parallel for
     for (int i = 0; i < N_f; i++) {
         double kappa2;
         while (1) {
             kappa2 = Random_powerlaw(gamma_f, k0_f);
             if (kappa2 <= kc_f && kappa2 >= k0_f) {
                 kappa_f[i] = kappa2;
                 break;
             }
         }
     }
 
     double *Theta_f = (double *) malloc(N_f * sizeof(double));
     #pragma omp parallel for
     for (int i = 0; i < N_f; i++) {
         double e = 2 * M_PI;
         Theta_f[i] = rand_easy(0, e);
     }
 
     R_bi = N_n_obs / (2 * M_PI * delta_n);
     mu_bi = (Beta_bi * sin(M_PI / Beta_bi)) / (2 * delta_n * kmean_n * M_PI);
 
     int it = 0;
     printf("Playing with Mu to get a Bipartite network with a desired average degree....!\n");
     double AVG_DEG_1, AVG_DEG_2;
     while (1)
     {
         printf("\n iteration # %d", it);
         printf("\n Mu is : %f", mu_bi);
         fp4 = fopen(Bi_Net_Add, "w+");
         if (fp4 == NULL) {
             perror("Failed: ");
             return 1;
         }
         double Num_links_bi = 0;
         #pragma omp parallel for reduction(+:Num_links_bi) schedule(dynamic)
         for (int ii = 0; ii < N_n; ii++) {
             for (int jj = 0; jj < N_f; jj++) {
                 double prob = Prob_Connection(kappa_n[ii], kappa_f[jj], Theta_n[ii], Theta_f[jj], Beta_bi, mu_bi, R_bi);
                 double r = rand_easy(0, 1);
                 if (r <= prob) {
                     #pragma omp critical
                     {
                         fprintf(fp4, "%d \t %d\n", ii, jj + N_n);
                     }
                     Num_links_bi += 1;
                 }
             }
         }
         fclose(fp4);
         AVG_DEG_1 = Num_links_bi / (double) N_n_obs;
         AVG_DEG_2 = Num_links_bi / (double) N_f_obs;
         printf("\n The average degree of nodes of Type 1 is: %f \n", AVG_DEG_1);
         printf(" The average degree of nodes of Type 2 is: %f \n", AVG_DEG_2);
         printf("~~~~~~~~~~~");
         if (fabs(AVG_DEG_1 - kmean_n) < 0.1 && fabs(AVG_DEG_2 - kmean_f) < 0.1)
             break;
         if (AVG_DEG_1 < kmean_n && AVG_DEG_2 < kmean_f)
             mu_bi = mu_bi + (0.01 * mu_bi);
         if (AVG_DEG_1 > kmean_n || AVG_DEG_2 > kmean_f)
             mu_bi = mu_bi - (0.005 * mu_bi);
         it++;
     }
 
     fp4 = fopen(Bi_Net_Add, "r");
     if (fp4 == NULL) {
         perror("Failed: ");
         return 1;
     }
     int NGZ1 = 0, NGZ2 = 0, Ver_num2, Edge_num2;
     read_file(Bi_Net_Add, &Ver_num2, &Edge_num2);
     int **Edgelist2 = (int **) malloc(Edge_num2 * sizeof(int *));
     for (int i = 0; i < Edge_num2; i++) {
         Edgelist2[i] = (int *) malloc(2 * sizeof(int));
     }
     int inx = 0;
     while (1)
     {
         if (feof(fp4))
             break;
         fscanf(fp4, "%d %d", &Edgelist2[inx][0], &Edgelist2[inx][1]);
         inx++;
     }
     fclose(fp4);
     for (int ii = 0; ii <= Ver_num2; ii++) {
         int found1 = 0, found2 = 0;
         for (int jj = 0; jj < Edge_num2; jj++) {
             if (Edgelist2[jj][0] == ii) {
                 found1 = 1;
                 break;
             }
             if (Edgelist2[jj][1] == ii) {
                 found2 = 1;
                 break;
             }
         }
         if (found1 == 1)
             NGZ1++;
         if (found2 == 1)
             NGZ2++;
     }
     printf("\n=========The properties of the generated bipartite networks=============\n");
     printf("The number of nodes of Type 1 with degree greater than zero is: %d\n", NGZ1);
     printf("The number of nodes of Type 2 with degree greater than zero is: %d\n", NGZ2);
     printf("The number of Edges is: %d\n", Edge_num2);
     double NGZ11 = NGZ1, NGZ22 = NGZ2;
     AVG_DEG_1 = Edge_num2 / NGZ11;
     AVG_DEG_2 = Edge_num2 / NGZ22;
     printf("The average degree of nodes of Type 1 is: %f \n", AVG_DEG_1);
     printf("The average degree of nodes of Type 2 is: %f \n", AVG_DEG_2);
 
     fp2 = fopen(Bi_Coordinate, "w+");
     if (fp2 == NULL) {
         perror("Failed: ");
         return 1;
     }
     for (int bb = 0; bb < N_n; bb++) {
         fprintf(fp2, "%d\t%f\t%f\n", bb, kappa_n[bb], Theta_n[bb]);
     }
     for (int bb = 0; bb < N_f; bb++) {
         fprintf(fp2, "%d\t%f\t%f\n", bb + N_n, kappa_f[bb], Theta_f[bb]);
     }
     fclose(fp2);
 
     free(New_Theta_s);
     free(New_kappa_s);
     free(kappa_n);
     free(Theta_n);
     free(kappa_f);
     free(Theta_f);
     for (int i = 0; i < Edge_num2; i++) {
         free(Edgelist2[i]);
     }
     free(Edgelist2);
 
 #ifdef _OPENMP
     free(thread_seeds);
 #endif
 
     return 0;
 }
 