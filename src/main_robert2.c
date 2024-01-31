//This code generates a unipartite network (S1) represents the structure, as well as, a bipartite network (S1*S1) shows the relations between nodes (Type1) and features (Type2)
//In the bipartite network, nodes of Type1  has the same Thetas as the unipartite. Their kappas are correlated with the kappas from the structural network as well
//The number of nodes of Type1 in the Bipartite network is equal to those of non-zero degrees in the unipartite network


// Run with `gcc -lm -O3 main.c`

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

double rand_easy(double a, double b); //Generate a uniformly distribute random number in [a, b] using rand()
double Prob_Connection(double kappa1, double kappa2, double tetha1, double tetha2, double B, double mu, double Rad); //it compute the connection probability of two nodes
double Random_powerlaw(double gamma, double k0);  //It picks a sample from a given power-law distribution
void read_file(const char *file_name , int * Nodes , int * Links); //It reads a file
double get_fun (double r, double k1, double k2, double k1_min, double k2_min , double gamma1, double gamma2, double nu); //This function computes the Eq.14 in the supplementary of "10.1038/NPHYS3812" paper to
                                                                                                                         //generate kappas in S1*S1 model which are correlated with those in the S1 model
double bisect (double r, double x1, double x2, double err,double k1, double k1_min, double k2_min , double gamma1, double gamma2, double nu); //Bisection method to find zeros of a function
double get_fun_truncated (double r, double k1, double k2, double k1_min, double k2_min , double k1_c, double k2_c, double gamma1, double gamma2, double nu);
double bisect_truncated (double r, double x1, double x2, double err,double k1, double k1_min, double k2_min, double k1_c, double k2_c, double gamma1, double gamma2, double nu);
void swap(int* xp, int* yp);
void selectionSort(int arr[], int n);
double Delta_Theta(double Theta1, double Theta2);
void Assign_Labels(double * Theta_s, int NC , int * Label, int NZ, float R, float alpha);

int main(int argc, char *argv[])
{
    /**
     * ./main [Beta_s] [gamma_s] [Ns_obs] [kmean_s] [gamma_n] [kmean_n] [gamma_f] [N_f] [Beta_bi] [nu] [alpha] [NC] [outfolder]
     *
     * S1 model parameters:
     *
     * Beta_s  - controls clustering
     * gamma_s - exponent of hidden degree distribution
     * Ns_obs  - size of network
     * kmean_s - average degree
     *
     *
     * S1xS1 model parameters:
     *
     * gamma_n - exponent of hidden degree distribution of Type 1 nodes (nodes)
     * kmean_n - average degree of Type 1 nodes
     * gamma_f - exponent of hidden degree distribution of Type 2 nodes (features)
     * N_f     - number of features
     * Beta_bi - Beta for bipartite model
     * nu      - correlation strength parameters
     * 
     * alpha   - exponent for compute the probability of assiging each node to a class
     * NC      - the number of labels/classes
     * outfolder - the folder to save the edgelist of networks
     *
     */
    if (argc != 14) {
        printf("Please provide all 13 parameters\n");
        exit(0);
    }


    //~~~~~~~~~~~~~~~~~~~~ Parameters of S1 model
    float Beta_s = atof(argv[1]);
    float gamma_s = atof(argv[2]);
    int Ns_obs = atoi(argv[3]); // The desired number of observed nodes in the resulting network
    int Ns;  //The number of nodes in the model
    float kmean_s = atof(argv[4]); //Average degree
    float Corr_k0_s= (1-(1/Ns_obs))/(1-pow(Ns_obs, ((2-gamma_s)/(gamma_s-1)))); //Correction of k0_s
    float k0_s = (((gamma_s-2) * kmean_s)/(gamma_s-1))* Corr_k0_s; //Minimum kappa
    float kc_s = k0_s * (pow(Ns_obs, (1/(gamma_s-1))));    //Maximum kappa
    int delta_s = 1;  //Density of nodes
    float R_s = Ns_obs / (2 * M_PI * delta_s); //The Radius of S1 model
    float mu_s= (Beta_s * sin(M_PI/Beta_s)) / (2 * delta_s * kmean_s * M_PI);  //controls the average degree
    float alpha_s = atof(argv[11]);
    int NC= atoi(argv[12]);

    //printf("alpha is %f\n", alpha_s);
    //printf("The number of classes is %d\n", NC);
    //~~~~~~~~~~~~~~~~~~~~ Parameters of S1_S1 model
    //~~~~Nodes of Type 1 (n)
    float gamma_n = atof(argv[5]);
    int N_n;
    int N_n_obs;     // number of nodes
    float kmean_n = atof(argv[6]);  // average degree of nodes of type 1 (Nodes)
    float Corr_k0_n; // Correction of k0_n
    float k0_n; // Minimum kappa of nodes of Type 1
    float kc_n; // Maximum kappa of nodes of Type 1
    int delta_n = 1;  //Density of nodes of type 1

    //~~~~Nodes of Type 2 (f)
    float gamma_f = atof(argv[7]);     //exponent of power-law distribution
    int N_f;
    int N_f_obs = atoi(argv[8]);    // number of features
    float kmean_f;  //average degree of nodes of type 2 (Features) kmeans_n * N_n = K_means_f * N_f
    float Corr_k0_f; //correction of k0_f
    float k0_f;   //Minimum kappa of nodes of Type 2
    float kc_f; //Maximum kappa of nodes of Type 2

    //~~~~
    float Beta_bi = atof(argv[9]);
    float R_bi ; //The Radius of S1*S1 model
    float mu_bi;  //controls the average degree

    //~~~~Parameter of correlation between kappas in S1 and kappas in S1*S1
    float Err = 0.001;  //Acceptable error of bisection method
    float nu = atof(argv[10]);   //nu in [0, 1] is the correlation strength parameter

    //~~~Set the seed for random number generator
    time_t t;
    srand((unsigned) time(&t));

    //~~~~~~~~~~~~~~~~~~~~~~~~~~Address and name of the resulting networks
    FILE * fp;
    FILE * fp2;
    FILE * fp3;
    FILE * fp4;

    char Node_s [33];
    sprintf(Node_s, "%d", Ns_obs);
    char B_s [33];
    sprintf (B_s, "%2.2f", Beta_s);
    char g_s [33];
    sprintf(g_s, "%2.2f", gamma_s);
    char K_s[33];
    sprintf(K_s, "%2.2f", kmean_s);
    char a_s[33];
    sprintf(a_s , "%2.2f", alpha_s);
    char num_c[33];
    sprintf(num_c, "%d", NC);



    // Create a folder with output edgelists
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

    char Node_1 [33];
    sprintf(Node_1, "%d", Ns_obs);
    char Node_2 [33];
    sprintf(Node_2, "%d", N_f_obs);
    char B_bi [33];
    sprintf (B_bi, "%2.2f", Beta_bi);
    char g1_bi [33];
    sprintf(g1_bi, "%2.2f", gamma_n);
    char g2_bi [33];
    sprintf(g2_bi, "%2.2f", gamma_f);
    char k1_bi[33];
    sprintf(k1_bi, "%2.2f", kmean_n);
    char nu_bi[33];
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

    char U_Coordinate[500] = "";       //This file keeps the coordinate of nodes in the unipartite network + Labels
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

    char Bi_Coordinate[500] = "";  //This file keeps the coordinate of nodes in the bipartite network
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

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~S1 Model~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //~~~~~~~~~~Parameter kappa/S1 Model    RA: I comment this part cause we prefer not to add extra node instead i have set Ns=Ns_obs

    //~~~~~~~~finding the number of nodes in the model (Ns) in order to have almost Ns_obs non-zero degree nodes in the resulting network
/*
    //double k_s;
    Ns = 0;
    while(1)
    {
        for (int i = Ns_obs; i<(2*Ns_obs); i++)
        {
            double P0=0;
            //k_s=0;
            int iter=0;
            float kappa;
            while (iter<i)       //Generating a bunch of size i of kappas where i is in [Ns_obs ... 2*Ns_obs]
            {
                 kappa = Random_powerlaw(gamma_s, k0_s);
                 if (kappa<=kc_s && kappa>=k0_s)
                 {
                     P0= P0 + exp(-kappa);
                     //k_s = k_s + kappa;
                     iter= iter+1;
                 }

            }
            P0 = P0 /i;
            //k_s= k_s/i;
           if (fabs( (i*(1-P0)) - Ns_obs) < 1)
           {
               Ns = i;
               //k_s = k_s /(1-P0);
               break;
           }

        }

        if (Ns > 0)
           break;

    }
    printf("\n The number of nodes in the S1 model is: %d \n", Ns);
   // printf("\n Their Average Kappa is: %f\n", k_s); */


    Ns=Ns_obs;

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Generating kappas
    double* kappa_s= (double*) malloc(Ns * sizeof(double));
    //  while(1)
    // {
       // double P0=0;
        int iter=0;
        double kappa;
        while(iter<Ns)
        {
            kappa=Random_powerlaw(gamma_s, k0_s);

            if(kappa<=kc_s && kappa>=k0_s)
            {
                kappa_s[iter]= kappa;
                //P0= P0 + exp(-kappa);
                iter=iter+1;
            }

        }
        //P0 = P0 /Ns;

        //if((Ns*(1-P0)) > (Ns_obs-1) &&  (Ns*(1-P0))< (Ns_obs+1))
        //{
          //  double out=  Ns*(1-P0);
          //  printf("\n The approximate number of nodes with degree greater than zero is : %f\n",out);
          //  break;
       // }
    //}
    //printf("==================================================================\n \n");

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Parameter Tetha/ S1 Model

    double* Theta_s= (double*) malloc(Ns * sizeof(double));

    for(int i=0; i<Ns; i++)
    {
       double e= 2* M_PI;
       Theta_s[i]=rand_easy(0, e);

    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Compute the connection probabilities and generate network of S1 Model
    int iteration=0;
    printf("Playing with Mu to get a unipartite network with a desired average degree....!\n");
    printf("=====================================\n");

    while(1)
    {
        printf("\n iteration # %d", iteration);
        printf("\n Mu is : %f", mu_s);

        // printf("\nfile = %s\n", S1_Net_Add);
        fp = fopen (S1_Net_Add, "w+");
        if (fp == NULL) {
            perror("Failed: ");
            return 1;
        }

        double Num_links=0;
        for(int ii=0; ii<(Ns-1) ; ii++) // RJ: why is here (Ns-1) ?? RA: in combination with the for loop below we go through all possible pairs (ii,jj) where ii~=jj
        {
            for(int jj= (ii+1); jj <Ns; jj++)
            {

                double prob = Prob_Connection(kappa_s[ii], kappa_s[jj], Theta_s[ii], Theta_s[jj], Beta_s, mu_s, R_s);
                double r= rand_easy(0, 1);
                if (r<= prob)
                {
                    fprintf(fp, "%d \t %d\n", ii, jj);
                    Num_links= Num_links+1;
                }

            }

        }

        fclose(fp);

        double AVG_DEG= 2* Num_links/Ns_obs;

        printf("\n The average degree of nodes is: %f \n", AVG_DEG);
        printf("~~~~~~~~~~~~~");

        if(fabs(AVG_DEG - kmean_s) < 0.1)
        {
            break;
        }

        // RJ: why you chose these values? maybe we could apply also bisection method here
        if(AVG_DEG<kmean_s)
        {
            // mu_s = mu_s+ (0.001*mu_s);
            mu_s = mu_s+ (0.01*mu_s);
        }
        if(AVG_DEG > kmean_s)
        {
            // mu_s = mu_s - (0.0005*mu_s) ;
            mu_s = mu_s - (0.005*mu_s) ;
        }

        iteration= iteration +1;
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Check the properties of the resulting network
    fp3= fopen (S1_Net_Add, "r");
    if (fp3 == NULL) {
        perror("Failed: ");
        return 1;
    }
    int NGZ=0;
    int Ver_num;
    int Edge_num;


    read_file(S1_Net_Add, &Ver_num, &Edge_num);


   int** Edgelist_s = (int**) malloc(Edge_num * sizeof(int*));
    for (int i=0; i< Edge_num ;  i++)
    {
        Edgelist_s[i]= (int*) malloc(2 * sizeof(int));
    }

    int ind=0;
    while (1) //read the edgelist
    {
        if (feof(fp3))
            break;

        fscanf(fp3, "%d %d", &Edgelist_s[ind][0] , &Edgelist_s[ind][1]);
        ind= ind+1;
    }
    fclose(fp3);

    printf("\n==============The properties of the generated unipartite networks==============\n");

    for(int ii=0; ii<=Ver_num; ii++)
    {
        int bool=0;

        for (int jj=0; jj<Edge_num; jj++)
        {
            if(Edgelist_s[jj][0]==ii || Edgelist_s[jj][1]==ii)
            {
                bool=1;
            }
        }

        if(bool==1)
        {
            NGZ= NGZ+1;
        }

    }

    printf("The number of nodes with degree greater than zero is: %d\n", NGZ);
    printf("The number of Edges is: %d\n", Edge_num);

    double NGZ0= NGZ;

    double AVG_DEG= 2*Edge_num/NGZ0;
    printf("The average degree of nodes is: %f", AVG_DEG);


    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Make a list of non-zero degree nodes and relabel them in a consecutive interval

    int* Nodelist_s= (int *) malloc(NGZ * sizeof (int));
    int* Nodelabel_s= (int *) malloc(NGZ * sizeof (int));
    for(int i=0; i < NGZ ;  i++)
    {
        Nodelabel_s[i]= i;
        Nodelist_s [i]= -1;
    }


    int iindd=0;
    for(int i=0; i<Edge_num; i++)
    {
        int Find0=0;
        for(int j=0; j<NGZ; j++)
        {
          if (Nodelist_s[j]==Edgelist_s[i][0])
          {
              Find0=1;
              break;
          }
        }

        if(Find0==0)
        {
            Nodelist_s[iindd]=Edgelist_s[i][0];
            iindd= iindd+1;
        }

    }

    for(int i=0; i<Edge_num; i++)
    {

        int Find0=0;
        for(int j=0; j<NGZ; j++)
        {

          if (Nodelist_s[j]==Edgelist_s[i][1])
          {
              Find0=1;
              break;
          }
        }

        if(Find0==0)
        {
            Nodelist_s[iindd]=Edgelist_s[i][1];
            iindd= iindd+1;
        }
    }

     selectionSort(Nodelist_s, NGZ);
     //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Rewrite the edgelist of the network based on new labels of nodes
     fp2 = fopen (S1_Net_Add, "w+");
     if (fp2 == NULL) {
         perror("Failed: ");
         return 1;
     }

     int temp0;
     int temp1;
     for(int ii=0; ii<Edge_num; ii++)
     {
         temp0=0;
         temp1=0;

         for (int jj=0; jj<NGZ; jj++)
         {
             if(Nodelist_s[jj]==Edgelist_s[ii][0])
             {
                 temp0=jj;
                 break;
             }

         }

         for (int kk=0; kk<NGZ; kk++)
         {
             if(Nodelist_s[kk]==Edgelist_s[ii][1])
             {
                 temp1=kk;
                 break;
             }

         }

         fprintf(fp2, "%d \t %d\n", Nodelabel_s[temp0], Nodelabel_s[temp1]);
     }
     fclose(fp2);

     //~~~~~~~~~~~~~~~~~~~Reorder Kappas and Thetas as well
     //~~~~~~~~~~~~~~~~~~~Assign labels to the nodes
     //~~~~~~~~~~~~~~~~~~Wirte the coordinate of nodes in a file in the format of [Node_ID] [kappa] [Theta] [Label]
     double* New_kappa_s= (double*) malloc(NGZ * sizeof(double));
     double* New_Theta_s= (double*) malloc(NGZ * sizeof (double));
     int* Label= (int*)malloc(NGZ * sizeof(int));


     for(int bb=0; bb<NGZ ;  bb++)
     {
         New_kappa_s[bb] =  kappa_s[Nodelist_s[bb]];
         New_Theta_s[bb] =  Theta_s[Nodelist_s[bb]];
     }

     Assign_Labels (New_Theta_s, NC, Label, NGZ, R_s, alpha_s);

     fp2 = fopen (U_Coordinate, "w+");
     if (fp2 == NULL) {
         perror("Failed: ");
         return 1;
     }

     for(int bb=0; bb<NGZ ;  bb++)
     {

         fprintf(fp2, "%d\t%f\t%f\t%d\n", bb, New_kappa_s[bb], New_Theta_s[bb], Label[bb]);
     }
     fclose(fp2);
     free(kappa_s);
     free(Theta_s);
     for (int i=0; i< Edge_num ;  i++)
     {
         free(Edgelist_s[i]);
     }

     free(Edgelist_s);
     free(Nodelabel_s);
     free(Nodelist_s);
     free(Label);

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ S1*S1 Model~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    printf("\n=========================Bipartite network=========================\n");
    //~~~~~~ Kappas of bipartite network for nodes of type 1 which are correlated with the kappas of S1 model

    N_n= NGZ;
    N_n_obs= NGZ;
    Corr_k0_n= (1-(1/N_n_obs))/(1-pow(N_n_obs, ((2-gamma_n)/(gamma_n-1))));
    k0_n = (((gamma_n-2) * kmean_n)/(gamma_n-1))* Corr_k0_n;
    kc_n = k0_n * (pow(N_n_obs, (1/(gamma_n-1))));
    double* kappa_n= (double*) malloc(N_n_obs * sizeof(double));
    //float kmean_n0;

    //~~~~~~~~~Intializing the starting and end points of bisection method
    double st_p = k0_n +0.000005 ;
    double end_p = kc_n ;

    double Rnd;
    //while (1)
    //{
        //kmean_n0 = 0;
        for (int jj=0; jj<N_n_obs; jj++)
        {
           kappa_n[jj]=0;
        }

        for (int ii=0; ii<N_n_obs; ii++)
        {
            double temp=nan("");

            Rnd = rand_easy(0, 1); //Random number between zero and one

            temp = bisect_truncated(Rnd, st_p, end_p, Err , New_kappa_s[ii], k0_s, k0_n, kc_s, kc_n,  gamma_s, gamma_n, nu);

            kappa_n[ii] = temp;
            //kmean_n0= kmean_n0 + temp;

        }

       // kmean_n0= kmean_n0 / N_n_obs ;

        printf("The number of nodes of Type1 is: %d\n", NGZ);
        printf("The desired average degree of nodes of Type1 is: %f\n", kmean_n);
       // printf("Average kappa of nodes of Type1 is: %f\n", kmean_n0);

        /*
        if(kmean_n0>kmean_n-0.1 && kmean_n0<kmean_n+0.1)
        {
            printf("Average kappa is: %f\n", kmean_n0);
            printf("<k_n> is: %f\n", kmean_n);
            break;
        }*/

   // }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Theta of bipartite network for node of Type 1
    double* Theta_n= (double*) malloc(N_n * sizeof(double));

    for(int i=0; i<N_n; i++)
    {
       Theta_n[i] = New_Theta_s[i];
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Kappas of bipartite network for nodes of Type 2
     kmean_f =  (kmean_n * N_n) / N_f_obs;
     Corr_k0_f= (1-(1/N_f_obs))/(1-pow(N_f_obs, ((2-gamma_f)/(gamma_f-1))));
     k0_f= (((gamma_f-2) * kmean_f)/(gamma_f-1)) * Corr_k0_f;
     kc_f = k0_f * (pow(N_f_obs, (1/(gamma_f-1))));
     N_f=0;
     double k_f;


    // RA: I comment this part cause we prefer not to add extra node instead i have set N_f=N_f_obs
    /* while (1)
     {
         for(int i = N_f_obs; i <( 2*N_f_obs) ; i++)
         {
            double P0=0;
            //k_f=0;
            int iter=0;
            float kappa;
            while(iter<i)
            {
                kappa=Random_powerlaw(gamma_f, k0_f);

               if(kappa<=kc_f && kappa>=k0_f)
               {
                   P0= P0 + exp(-kappa);
                   //k_f = k_f + kappa;
                   iter= iter+1;
               }
            }
            P0 = P0 /i;
            //k_f= k_f/i;
            if((i*(1-P0)) > (N_f_obs-0.5) &&  (i*(1-P0))< (N_f_obs+0.5))
            {
                N_f = i;
               // k_f = k_f /(1-P0);
                break;
            }

        }

       if(N_f >0)
          break;

     }*/
     N_f= N_f_obs;


     printf("\nThe number of nodes of Type2 is: %d \n", N_f);
     printf("Their desired average degree of nodes of Type2 is: %f\n", kmean_f);
     //printf("\n Average kappa of nodes of Type2 is: %f\n", k_f);

     //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     double* kappa_f= (double*) malloc(N_f * sizeof(double));
     //while(1)
    // {
       //  double P0=0;
         //k_f=0;
         int iter2=0;
         float kappa2;
         while(iter2<N_f)
         {
             kappa2=Random_powerlaw(gamma_f, k0_f);

             if(kappa2 <= kc_f && kappa2>=k0_f)
             {
                 kappa_f[iter2]= kappa2;
                 //P0= P0 + exp(-kappa);
                // k_f = k_f + kappa;
                 iter2= iter2+1;

             }
         }
         //P0 = P0 /N_f;
        // k_f= k_f/N_f;

        // RJ: if (fabs(N_f_obs - (N_f * (1 - P0))) < 0.5)
         //if((N_f*(1-P0)) > (N_f_obs-0.5) &&  (N_f*(1-P0))< (N_f_obs+0.5))// && k_f /(1-P0)>(kmean_f-0.1) && k_f /(1-P0)<(kmean_f+0.1) )
         //{
         //    //k_f = k_f /(1-P0);
         //    double out=  N_f*(1-P0);
          //   printf("\nThe approximate number of non-zero degree nodes of Type2 is: %f\n",out);
         //    break;
         //}
    // }
     printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
    //~~~~~~~~~~~~~~~~~~~Thetas of bipartite network for nodes of Type 2
    double* Theta_f= (double*) malloc(N_f * sizeof(double)); //Tetta/Features for nodes of type 2
    for(int i=0; i<N_f; i++)
    {
       double e= 2* M_PI;
       Theta_f[i]=rand_easy(0, e);
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Compute the probability of connection between nodes in S1 * S1 and generate the bipartite network

    R_bi = N_n_obs / (2 * M_PI * delta_n); //The Radius of S1_S1 model
    mu_bi= (Beta_bi * sin(M_PI/Beta_bi)) / (2 * delta_n * kmean_n * M_PI);

    int it=0;
    printf("Playing with Mu to get a Bipartite network with a desired average degree....!\n");
    while(1)
    {
        printf("\n iteration # %d", it);
        printf("\n Mu is : %f", mu_bi);

        fp4 = fopen (Bi_Net_Add, "w+");
        if (fp4 == NULL) {
            perror("Failed: ");
            return 1;
        }
        double Num_links=0;

        for(int ii=0; ii<N_n ; ii++)
        {
            for(int jj=0 ;jj<N_f; jj++)
            {
                double prob = Prob_Connection(kappa_n[ii], kappa_f[jj], Theta_n[ii], Theta_f[jj], Beta_bi, mu_bi, R_bi);
                double r= rand_easy(0, 1);
                if (r<= prob)
                {
                    fprintf(fp4, "%d \t %d\n", ii, jj+N_n);
                    Num_links= Num_links+1;
                }
            }

        }

        fclose(fp4);
        double AVG_DEG_1= Num_links/N_n_obs;
        double AVG_DEG_2= Num_links/N_f_obs;
        printf("\n The average degree of nodes of Type 1 is: %f \n", AVG_DEG_1);
        printf(" The average degree of nodes of Type 2 is: %f \n", AVG_DEG_2);
        printf("~~~~~~~~~~~");

        if(fabs(AVG_DEG_1-kmean_n) < 0.1 && fabs(AVG_DEG_2-kmean_f) <0.1)
        {
            break;
        }

        if(AVG_DEG_1<kmean_n && AVG_DEG_2<kmean_f)
        {
            // mu_bi = mu_bi+ (0.001*mu_bi);
            mu_bi = mu_bi+ (0.01*mu_bi);
        }
        if(AVG_DEG_1 > kmean_n || AVG_DEG_2>kmean_f)
        {
            // mu_bi = mu_bi - (0.0005*mu_bi);
            mu_bi = mu_bi - (0.005*mu_bi);
        }

        it= it +1;
    }
     //**********************************************Check the properties of the resulting network
    fp4= fopen (Bi_Net_Add, "r");
    if (fp4 == NULL) {
        perror("Failed: ");
        return 1;
    }
    int NGZ1=0;
    int NGZ2=0;

    int Ver_num2;
    int Edge_num2;

    read_file(Bi_Net_Add, &Ver_num2, &Edge_num2);

    int** Edgelist2 = (int**) malloc(Edge_num2 * sizeof(int*));
    for (int i=0; i< Edge_num2 ;  i++)
    {
        Edgelist2[i]= (int*) malloc(2 * sizeof(int));
    }

    int inx=0;
    while (1) //read the edgelist
    {
        if (feof(fp4))
            break;

        fscanf(fp4, "%d %d", &Edgelist2[inx][0], &Edgelist2[inx][1]);
        inx= inx+1;
    }
    fclose(fp4);

    for(int ii=0; ii<=Ver_num2; ii++)
    {
        int bool=0;
        int bool2=0;
        for (int jj=0; jj<Edge_num2; jj++)
        {
            if(Edgelist2[jj][0]==ii)
            {
                bool=1;
                  break;
            }
            if(Edgelist2[jj][1]==ii)
            {
                bool2=1;
                 break;
            }
        }
        if(bool==1)
        {
            NGZ1= NGZ1+1;
        }
        if(bool2==1)
        {
            NGZ2= NGZ2+1;
        }
    }
    printf("\n=========The properties of the generated unipartite networks=============\n");
    printf("The number of nodes of Type 1 with degree greater than zero is: %d\n", NGZ1);
    printf("The number of nodes of Type 2 with degree greater than zero is: %d\n", NGZ2);
    printf("The number of Edges is: %d\n", Edge_num2);

    double NGZ11= NGZ1;
    double NGZ22= NGZ2;

    double AVG_DEG_1= Edge_num2/NGZ11;
    double AVG_DEG_2= Edge_num2/NGZ22;
    printf("The average degree of nodes of Type 1 is: %f \n", AVG_DEG_1);
    printf("The average degree of nodes of Type 2 is: %f \n", AVG_DEG_2);

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~Write the coordinate of Bipartite network
    fp2 = fopen (Bi_Coordinate, "w+");
    if (fp2 == NULL) {
        perror("Failed: ");
        return 1;
    }

    for(int bb=0; bb<N_n ;  bb++)
    {
        fprintf(fp2, "%d\t%f\t%f\n", bb, kappa_n[bb], Theta_n[bb]);
    }
    for(int bb=0; bb<N_f ;  bb++)
    {
        fprintf(fp2, "%d\t%f\t%f\n", bb+N_n, kappa_f[bb], Theta_f[bb]);
    }

    fclose(fp2);


    free(New_Theta_s);
    free(New_kappa_s);
    free(kappa_n);
    free(Theta_n);
    free(kappa_f);
    free(Theta_f);

    for (int i=0; i< Edge_num2 ;  i++)
    {
        free(Edgelist2[i]);
    }
    free(Edgelist2);

}
double rand_easy(double a, double b) //Generate a uniformly distribute random number in [a, b] using rand() that generates a number between [0, RAND_Max]
{
    double range= b - a;
    return ((((double)rand() / (double)RAND_MAX) * range) + a);
}

double Prob_Connection(double kappa1, double kappa2, double tetha1, double tetha2, double B, double mu, double Rad) //it compute the connection probability of two nodes
{
    double Delta_T= M_PI - fabs(M_PI- fabs(tetha1- tetha2));
    double X= (Rad * Delta_T)/(mu * kappa1 * kappa2);
    double result = 1 / (1 + pow(X, B));
    return(result);
}

double Random_powerlaw(double gamma, double k0)
{
    double prob= rand_easy(0, 1); //Select a random number between 0 and 1
    double value = k0 / (pow ((1-prob), (1/(gamma-1))));  //CCDF of a power law degree distribution
    return value;
}

void read_file(const char *file_name , int * Nodes , int * Links) {
    FILE *myfile = fopen(file_name, "r");
    int Num_Nodes = 0;
    int Num_links = 0;
    int Node1;
    int Node2;

    while (1) //Find the number of vertices
    {
        fscanf(myfile, "%d %d", &Node1 , &Node2);

        if (feof(myfile))
            break;

        if (Node1>Num_Nodes)
        {
            Num_Nodes= Node1;
        }

        if (Node2>Num_Nodes)
        {
            Num_Nodes= Node2;
        }
        Num_links = Num_links+1;

     }

    * Nodes = Num_Nodes ;
    * Links = Num_links;
    fclose(myfile);
}

double get_fun (double r, double k1, double k2, double k1_min, double k2_min , double gamma1, double gamma2, double nu)
{
    double phi1, phi2, C, res, temp;
    phi1 =  -log (1-(pow((k1_min/k1), (gamma1-1))));
    phi2 =  -log (1- (pow((k2_min/k2), (gamma2-1))));
    C= (pow(phi1,(nu/(1-nu))) * k1_min * pow(k1, gamma1)) / ((k1_min * pow(k1, gamma1)) - (pow(k1_min, gamma1) * k1));
    temp= pow(phi1, (1/(1-nu))) + pow(phi2, (1/(1-nu)));
    res = (exp(-(pow(temp, (1-nu)))) * pow(temp, -nu) * C) - r ;
    return(res);
}

double get_fun_truncated (double r, double k1, double k2, double k1_min, double k2_min , double k1_c, double k2_c, double gamma1, double gamma2, double nu)
{
    double phi1, phi2, C, res, temp;
    phi1 =  -log (1-((pow(k1, (1-gamma1)) - (pow(k1_c, (1-gamma1)))) / ((pow(k1_min, (1-gamma1)) - (pow(k1_c, (1-gamma1)))))));
    phi2 =  -log (1-((pow(k2, (1-gamma2)) - (pow(k2_c, (1-gamma2)))) / ((pow(k2_min, (1-gamma2)) - (pow(k2_c, (1-gamma2)))))));
    C=  pow(phi1,(nu/(1-nu))) * (1 / (1-((pow(k1, (1-gamma1)) - (pow(k1_c, (1-gamma1)))) / ((pow(k1_min, (1-gamma1)) - (pow(k1_c, (1-gamma1))))))));
    temp= pow(phi1, (1/(1-nu))) + pow(phi2, (1/(1-nu)));
    res = (exp(-(pow(temp, (1-nu)))) * pow(temp, -nu) * C) - r ;
    return(res);
}


double bisect (double r, double x1, double x2, double err,double k1, double k1_min, double k2_min , double gamma1, double gamma2, double nu)
{
   double f1, f2, f;
   double x;
   int iter=0;
   while(1)
  {
      f1= get_fun(r, k1, x1, k1_min,k2_min,  gamma1,  gamma2, nu);
      f2= get_fun(r, k1, x2, k1_min,k2_min,  gamma1,  gamma2, nu);
      if(f1 * f2 >0 || isnan(f1) || isnan(f2))
      {
          printf("\n initial guesses are wrong!\n");
          double T=nan("");
          return T;   //it retruns nan
      }

      x= (x1+x2)/2;
      if(((x2-x1)/x) < err)
      {
          f=get_fun( r, k1, x, k1_min,  k2_min ,  gamma1,  gamma2, nu);
          return x;
      }

      f=get_fun( r, k1, x, k1_min,  k2_min ,  gamma1,  gamma2, nu);

      if( (f*f1)>0)
      {
          x1 = x;
          f1=f;
      }
      else
      {
          x2= x;
          f2 = f;
      }
      iter= iter+1;
   }
}

double bisect_truncated (double r, double x1, double x2, double err,double k1, double k1_min, double k2_min, double k1_c, double k2_c, double gamma1, double gamma2, double nu)
{
   double f1, f2, f;
   double x;
   int iter=0;

   f1= get_fun_truncated(r, k1, x1, k1_min, k2_min, k1_c, k2_c,  gamma1,  gamma2, nu);
   f2= get_fun_truncated(r, k1, x2, k1_min,k2_min, k1_c, k2_c, gamma1,  gamma2, nu);
   if(f1 * f2 >0)
   {
       printf("%f \t %f \n", f1, f2);

       while(1)
       {
           //printf("\nchecking....\n");
           x1= rand_easy(k2_min, k2_c);
           x2= rand_easy(k2_min, k2_c);

           f1= get_fun_truncated(r, k1, x1, k1_min, k2_min, k1_c, k2_c,  gamma1,  gamma2, nu);
           f2= get_fun_truncated(r, k1, x2, k1_min,k2_min, k1_c, k2_c, gamma1,  gamma2, nu);

           if(f1*f2<0)
           {
               printf("******Yes!!********\n");
               printf("%f \t %f \n", f1, f2);
               break;
           }
       }
   }
    while(1)
   {
        f1= get_fun_truncated(r, k1, x1, k1_min, k2_min, k1_c, k2_c,  gamma1,  gamma2, nu);
        f2= get_fun_truncated(r, k1, x2, k1_min,k2_min, k1_c, k2_c, gamma1,  gamma2, nu);
         x= (x1+x2)/2;
         if(((x2-x1)/x) < err)
         {
             f=get_fun_truncated( r, k1, x, k1_min,  k2_min , k1_c, k2_c, gamma1,  gamma2, nu);
              return x;
          }

          f=get_fun_truncated( r, k1, x, k1_min,  k2_min, k1_c, k2_c,  gamma1,  gamma2, nu);

          if( (f*f1)>0)
          {
              x1 = x;
              f1=f;
          }
          else
          {
              x2= x;
              f2 = f;
          }
          iter= iter+1;
   }
}



void swap(int* xp, int* yp)
{
    int temp = *xp;
    *xp = *yp;
    *yp = temp;
}
// Function to perform Selection Sort
void selectionSort(int arr[], int n)
{
    int i, j, min_idx;

    // One by one move boundary of unsorted subarray
    for (i = 0; i < n - 1; i++) {

        // Find the minimum element in unsorted array
        min_idx = i;
        for (j = i + 1; j < n; j++)
            if (arr[j] < arr[min_idx])
                min_idx = j;

        // Swap the found minimum element
        // with the first element
        swap(&arr[min_idx], &arr[i]);
    }
}

double Delta_Theta(double Theta1, double Theta2) //Generate a uniformly distribute random number in [a, b] using rand() that generates a number between [0, RAND_Max]
{
    return (M_PI - fabs(M_PI- fabs(Theta1 - Theta2)));
}


void Assign_Labels(double * Theta_s, int NC , int * Label, int NZ, float R, float alpha)
{
    float c_centers[NC];
    double e= 2* M_PI;

    float r;

    alpha= -1 * alpha;

    printf("\n");
    for (int i=0; i< NC ; i++)     //choose centers at random
    {
        c_centers[i]= rand_easy(0, e);
        printf("Center number %d is located at: %f\n", i, c_centers[i]);

    }

    for (int node=0; node<NZ ; node++)    //for every node
    {
        float d[NC];
        float pr[NC];
        float delta_Th;
        float d_sum;
        d_sum=0;
        for(int l=0; l<NC; l++)        //for every center
        {

            delta_Th= Delta_Theta(Theta_s[node],c_centers[l]);     //compute delta tetha between node and center
            d[l]= R * delta_Th;            //compute distance
            d_sum= d_sum + pow(d[l], alpha);

        }

        for(int l=0; l<NC; l++)     //compute the probability of assigning a node to each of classes (pr keeps the comulative probability)
        {
            if (l==0)
                pr[l]= pow(d[l], alpha) / d_sum;
            else
                pr[l]= (pow(d[l], alpha) / d_sum) + pr[l-1];


        }

        r= rand_easy(0, 1);    //choose a random number and assign label regarding this number

        for(int l=0; l<NC; l++)
        {
            if(l==0)
            {
                if (r>=0 && r<pr[l])
                {
                    Label[node]= l;
                    break;
                }
            }
            else
            {
                if (r>= pr[l-1] && r<pr[l])
                {
                    Label[node]=l;
                    break;
                }

            }
        }


    }



}
