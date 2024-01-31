#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>


/*
 This code is run with ./main.out  [Edgelist file address of Unipartite network] [Coordinate file address of Unipartite network] [mu] [Beta of unipartite network] [New_Beta]
*/


void read_file(const char *file_name , int * Nodes , int *Links);
double rand_easy(double a, double b);
double Delta_Theta(double Theta1, double Theta2);
double Prob_Connection(double kappa1, double kappa2, double theta1, double theta2, double B, double mu, double Rad);
long double log_likelihood_init(int** Edge_list , int Num_Nodes, int Num_Links , double * Theta, double * kappa, double Beta, double mu, double R);
long double update_log_likelihood(long double likelihood, float Beta,  double Delta_Theta_ij, double Delta_Theta_lm, double Delta_Theta_im, double Delta_Theta_lj);
void Log_likelihood_Plateau(double log_likelihood_Network, int N_Links, int ** Edgelist, double * Theta, float Beta, FILE * fp, float threshold);
void Geometric_Randomization(int ** Edgelist, int Num_stepp, int N_Links, double * Theta, float Beta);
int main(int argc, char* argv[])
{
    if (argc != 6) {
        printf("Please provide 5 inputs \n");
        exit(0);
    }

    //~~~Set the seed for random number generator
        time_t t;
        srand((unsigned) time(&t));
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    char *Net_add = argv[1];
    printf("Edgelist file is: %s\n\n", Net_add);

    char *Coord_add= argv[2];
    printf("Coordinate file is: %s\n\n", Coord_add);

    float mu= atof(argv[3]);
    printf("mu is: %f\n", mu);

    float Beta= atof(argv[4]);

    //~~~~Extract folder address from the address of coordinate file
    char *lastSlash = NULL;

    char *folder_name = strdup(Coord_add); // Create a duplicate of Coord_add
    lastSlash = strrchr(folder_name, '/');
    if (lastSlash != NULL) {
        *(lastSlash + 1) = '\0'; // Set the character after the last slash to null terminator
    }

    char *folder_name2=malloc(strlen(folder_name) + strlen("loglikelihood.txt") + 1);
    strcpy(folder_name2, folder_name);

    float New_Beta= atof(argv[5]);
    int N_Nodes;
    int N_Links;
    FILE* fp;

    //~~~~~~~~~Read Edgelist

    read_file(Net_add, &N_Nodes , &N_Links);

    printf("The number of nodes is: %d \n", N_Nodes);
    printf("The number of links is: %d \n", N_Links);


    float R = N_Nodes/ (2 * M_PI ); //The Radius of S1 model

    fp = fopen (Net_add, "r");
    if (fp == NULL)
    {
        printf("Failed to open the file.\n");
        return 1;
    }

    int** Edgelist_init = (int**) malloc(N_Links * sizeof(int*));
    for (int i=0; i< N_Links ;  i++)
    {
        Edgelist_init[i]= (int*) malloc(2 * sizeof(int));
    }

    int id=0;
    while (id<N_Links) //Find the number of vertices
    {
        if (feof(fp))
            break;

        if(fscanf(fp, "%d %d", &Edgelist_init[id][0] , &Edgelist_init[id][1])!=2)
        {

            printf("Error in reading from a file 2 \n");
            return 1;
        }

        id= id+1;
    }
    fclose(fp);
    //~~~~~~~~~~Read Coordinates
    double * kappa = (double *) malloc(N_Nodes * sizeof(double));
    double * Theta= (double*) malloc(N_Nodes * sizeof(double));
    fp = fopen (Coord_add, "r");
    if (fp == NULL)
    {
        printf("Failed to open the file.\n");
        return 1;
    }
    int id2=0;

    while(id2<N_Nodes)
    {
        if(feof(fp))
            break;
        if (fscanf(fp, "%*d %lf %lf %*d", & kappa[id2] , & Theta[id2])!=2)
        {
            printf("Error in reading from a file 3 \n");
            return 1;
        }
        //printf("%f \t %f\n", kappa[id2],  Theta[id2]);
        id2 = id2 +1;
    }
    fclose(fp);
    //~~~~~~~~~~~~~~~~~~~~~~Compute the log-likelihood of the network before generating randomized network and randmoize with New Beta to reach to the plateau
    long double log_likelihood_Network;
    //log_likelihood_Network = log_likelihood_init(Edgelist_init , N_Nodes, N_Links , Theta,  kappa , Beta,  mu, R);
    log_likelihood_Network= -100000; //We can find the exact log-likelihood but in some cases it is time consuming and as we only update it relatively we can set it to a arbitrary value

    printf("The initial log likelihood is: %Lf\n", log_likelihood_Network);
    float threshold =0.01;     //Acceptable error to check if we reach the plateau
    FILE * fp1;
    strcat(folder_name2, "loglikelihood.txt");

    fp1= fopen(folder_name2, "w+");
    Log_likelihood_Plateau(log_likelihood_Network, N_Links, Edgelist_init, Theta, Beta, fp1, threshold);


    //~~~~~~~~~~~~~~~~~~~~~~~~~~

    int Num_step= 10 * N_Links;    //Number of time that we do the link swap


    Geometric_Randomization( Edgelist_init,  Num_step,  N_Links, Theta,  New_Beta);

    char snum0 [30];
    sprintf (snum0, "%.2f", New_Beta);
    strcat(folder_name, "GR_");
    strcat(folder_name, "B_");
    strcat(folder_name, snum0 );
    strcat(folder_name, ".unipartite.edgelist.rand");

    //printf("%s\n", folder_name);
    FILE * fp2 = fopen(folder_name,  "w+");
    for (int i = 0; i < N_Links; i++)
    {
        fprintf(fp2, "%d\t%d\n" , Edgelist_init[i][0], Edgelist_init[i][1]);

    }
    fclose(fp2);

    for (int i=0; i< N_Links ;  i++)
    {
        free(Edgelist_init[i]);
    }

    free(Edgelist_init);
    free(kappa);
    free(Theta);
    return 0;
}
//-------------------------------------------------------------------------------------------
//        Function to read the edgelist and compute the number of nodes and edges
//-------------------------------------------------------------------------------------------
void read_file(const char *file_name , int * Nodes , int *Links) {
    FILE *myfile = fopen(file_name, "r");
    int NN = 0; //Number of Nodes
    int NL = 0; //Number of Links
    int N1;  //id of Node 1 in the rows of Edgelist
    int N2;  //id of Node 2 in  the rows of Edgelist

    while (1) //Find the number of vertices
    {
        fscanf(myfile, "%d %d", &N1 , &N2);

        if (feof(myfile))
            break;
        if (N1>NN)
        {
            NN= N1;
        }
        if (N2>NN)
        {
            NN= N2;
        }
        NL = NL+1;
        //printf("%d\n", NL);

     }
    * Nodes = NN +1 ;
    * Links = NL;
    fclose(myfile);
}


//-------------------------------------------------------------------------------------------
//        Function to generate a uniformly distribute random number in [a, b]
//-------------------------------------------------------------------------------------------
double rand_easy(double a, double b)
{
    double range= b - a;
    return ((((double)rand() / (double)RAND_MAX) * range) + a);
}
//-------------------------------------------------------------------------------------------
//        Function to compute Delta Theta between two nodes
//-------------------------------------------------------------------------------------------

double Delta_Theta(double Theta1, double Theta2)
{
    return (M_PI - fabs(M_PI- fabs(Theta1 - Theta2)));
}

//-------------------------------------------------------------------------------------------
//        Function to compute probability of connection between two nodes
//-------------------------------------------------------------------------------------------
double Prob_Connection(double kappa1, double kappa2, double theta1, double theta2, double B, double mu, double Rad) //it compute the connection probability of two nodes
{
    double Delta_T=Delta_Theta(theta1, theta2);
    double X= (Rad * Delta_T)/(mu * kappa1 * kappa2);
    double result = 1 / (1 + pow(X, B));
    return(result);
}
//-------------------------------------------------------------------------------------------
//        Function to compute the initial log likelihood of the network being generated by S1 model
//-------------------------------------------------------------------------------------------
long double log_likelihood_init(int** Edge_list , int Num_Nodes, int Num_Links , double * Theta, double * kappa, double Beta, double mu, double R)
{
    long double temp=0;

    long double log_likelihood=0;
    double pij;
    for (int i=0; i< (Num_Nodes-1) ;  i++)
    {
        for(int j=i+1; j<Num_Nodes; j++)
        {
            int a=0;
            for (int k=0; k<Num_Links ; k++)
            {
                if((Edge_list[k][0]==i && Edge_list[k][1]==j) || (Edge_list[k][0]==j && Edge_list[k][1]==i))
                {
                    a=1;
                    break;
                }
             }

            pij= Prob_Connection(kappa[i], kappa[j], Theta[i], Theta[j], Beta, mu, R);

            if (pij>0 && pij<1)
            {
               temp=  (a * log (pij)) + ((1-a) * log(1-pij));
               //printf("%Lf\n", temp);
               log_likelihood = log_likelihood + temp;
               //printf("%Lf **********\n", log_likelihood);
            }

        }
    }
    //printf("\n We have computed log likelihood==============================\n");
    return (log_likelihood);
}

//-------------------------------------------------------------------------------------------
//        Function to update log-likelihood after each link swap
//-------------------------------------------------------------------------------------------

long double update_log_likelihood(long double likelihood, float Beta,  double Delta_Theta_ij, double Delta_Theta_lm, double Delta_Theta_im, double Delta_Theta_lj)
{

    double temp = 0;
    long double New_likelihood=likelihood;
    temp= Beta  * (log(Delta_Theta_ij)+ log(Delta_Theta_lm) - log(Delta_Theta_im) - log(Delta_Theta_lj));
    New_likelihood =  New_likelihood + temp;
   // printf("New log likelihood in the fucntion is %Lf\n", New_likelihood);
    return (New_likelihood);
}

//-------------------------------------------------------------------------------------------
//        Function to compute the initial log likelihood of the network being generated by S1 model
//-------------------------------------------------------------------------------------------
void Log_likelihood_Plateau(double log_likelihood_Network, int N_Links, int ** Edgelist, double * Theta, float Beta, FILE * fp, float threshold)
{
    fprintf(fp, "%lf \n", log_likelihood_Network);
    long long int steps = 0;
    int rewired=0;
    int slope_range_x =10000;
    long double slope;

    struct likelihood {
        double value;
        struct likelihood* next;
    };

    struct likelihood *log_likelihood=NULL;
    struct likelihood *temp;

    //long double log_likelihood_vec[10000]={0};
    long double temppp=0;

    while (1)
    {
        if (log_likelihood == NULL)
        {
            temp = (struct likelihood*)malloc(sizeof(struct likelihood));
            temp->value =log_likelihood_Network;
            temp->next = NULL;
            log_likelihood= temp;

        }

        if ((steps % 100000)==0)
        {
           printf("It is the step number %llu\n", steps);
           printf("Slope is %Lf \n", slope);
           printf("rewired is %d\n", rewired);
           printf("The network log likelihood is %lf \n", log_likelihood_Network);
        }


        int ind1= rand() % N_Links;
        int ind2= rand() % N_Links;

        int nodei = Edgelist[ind1][0];
        int nodej= Edgelist[ind1][1];

        //printf("node i is %d \n", nodei);
        //printf("node j is %d \n", nodej);

        int nodel = Edgelist[ind2][0];
        int nodem = Edgelist[ind2][1];

        double Delta_Theta_ij= Delta_Theta(Theta[nodei], Theta[nodej]);
        double Delta_Theta_lm = Delta_Theta(Theta[nodel], Theta[nodem]);
        double Delta_Theta_im= Delta_Theta(Theta[nodei], Theta[nodem]);
        double Delta_Theta_lj= Delta_Theta(Theta[nodel], Theta[nodej]);

        long double likelihood_Ratio = pow(((Delta_Theta_ij * Delta_Theta_lm) / (Delta_Theta_im * Delta_Theta_lj)), Beta);
        int Bl1= 1;
        int Bl2 = 1;
        int Bl3 = 1;
        for (int i=0; i< N_Links ; i ++)
        {
            if( (Edgelist[i][0]==Edgelist[ind1][0] && Edgelist[i][1] == Edgelist[ind2][1]) || (Edgelist[i][1]==Edgelist[ind1][0] && Edgelist[i][0] == Edgelist[ind2][1]))   //Recreate an existing link
            {
                Bl1= 0;

            }

            if( (Edgelist[i][0]==Edgelist[ind2][0] && Edgelist[i][1] == Edgelist[ind1][1]) || (Edgelist[i][1]==Edgelist[ind2][0] && Edgelist[i][0] == Edgelist[ind1][1]))   //Recreate an existing link
            {
                Bl2=0;
            }
            if( (Edgelist[ind1][1]==Edgelist[ind2][0]) || (Edgelist[ind1][0]==Edgelist[ind2][1]))  //Create a self loop
            {
                Bl3=0;
            }

        }

        double  pr= likelihood_Ratio;
        double r= rand_easy(0, 1);
        if ((Bl1 * Bl2 * Bl3) ==1 && (pr>= 1 || r<= pr)  )
        {
            int Temp1= Edgelist[ind1][1];
            Edgelist[ind1][1]= Edgelist[ind2][1];
            Edgelist[ind2][1] = Temp1;

            temppp = update_log_likelihood(log_likelihood_Network, Beta, Delta_Theta_ij, Delta_Theta_lm, Delta_Theta_im, Delta_Theta_lj);
            log_likelihood_Network = temppp;
            //printf("HI\n");

            fprintf(fp, "%lf \n", log_likelihood_Network);
            //printf("%Lf \n", log_likelihood_Network);

             rewired= rewired+1;
             temp = (struct likelihood*)malloc(sizeof(struct likelihood));
             temp->value =log_likelihood_Network;
             temp->next= NULL;

             struct likelihood *lastNode = log_likelihood;
             //last node's next address will be NULL.
             while(lastNode->next != NULL)
             {
                 lastNode = lastNode->next;
             }

             lastNode->next=temp;

             if (rewired > slope_range_x)    //we oy keep the last #slope_range_x values
             {
                 log_likelihood= log_likelihood->next;
             }

        }

        steps=steps+1;
        int length_linked_list=0;

        if (rewired > slope_range_x)
        {
            double llh_first= 0.0;
            double llh_end= 0.0;

            llh_first = log_likelihood->value;
            struct likelihood *lastNode = log_likelihood;
            //last node's next address will be NULL.
            while(lastNode->next != NULL)
            {
                lastNode = lastNode->next;
                length_linked_list=length_linked_list+1;
            }
            llh_end= lastNode->value;

            slope = (llh_end - llh_first)/ slope_range_x;
            //printf("Slope is %Lf \n", slope);
            if (fabsl(slope) <threshold)
            {
                printf("Slope is %Lf \n", slope);
                break;
            }
             //printf("length of linked list is %d \n", length_linked_list);
        }



    }
    fclose(fp);
}

//-------------------------------------------------------------------------------------------
//        Function to do Geometric_Randomization
//-------------------------------------------------------------------------------------------
void Geometric_Randomization(int ** Edgelist, int Num_stepp, int N_Links, double * Theta, float Beta)
{
    int cc=0;
    while(1)
    {
        if(cc>Num_stepp)
        {
            break;
        }

        int ind1= rand() % N_Links;
        int ind2= rand() % N_Links;

        int nodei = Edgelist[ind1][0];
        int nodej= Edgelist[ind1][1];

        int nodel = Edgelist[ind2][0];
        int nodem = Edgelist[ind2][1];

        double Delta_Theta_ij= Delta_Theta(Theta[nodei], Theta[nodej]);
        double Delta_Theta_lm = Delta_Theta(Theta[nodel], Theta[nodem]);
        double Delta_Theta_im= Delta_Theta(Theta[nodei], Theta[nodem]);
        double Delta_Theta_lj= Delta_Theta(Theta[nodel], Theta[nodej]);

        long double likelihood_Ratio = pow(((Delta_Theta_ij * Delta_Theta_lm) / (Delta_Theta_im * Delta_Theta_lj)), Beta);
        int Bl1= 1;
        int Bl2 = 1;
        int Bl3 = 1;
        for (int i=0; i< N_Links ; i ++)
        {
            if( (Edgelist[i][0]==Edgelist[ind1][0] && Edgelist[i][1] == Edgelist[ind2][1]) || (Edgelist[i][1]==Edgelist[ind1][0] && Edgelist[i][0] == Edgelist[ind2][1]))   //Recreate an existing link
            {
                Bl1= 0;

            }

            if( (Edgelist[i][0]==Edgelist[ind2][0] && Edgelist[i][1] == Edgelist[ind1][1]) || (Edgelist[i][1]==Edgelist[ind2][0] && Edgelist[i][0] == Edgelist[ind1][1]))   //Recreate an existing link
            {
                Bl2=0;
            }
            if( (Edgelist[ind1][1]==Edgelist[ind2][0]) || (Edgelist[ind1][0]==Edgelist[ind2][1]))  //Create a self loop
            {
                Bl3=0;
            }

        }

        double  pr= likelihood_Ratio;
        double r= rand_easy(0, 1);
        if ((Bl1 * Bl2 * Bl3) ==1 && (pr>= 1 || r<= pr)  )
        {
            int Temp1= Edgelist[ind1][1];
            Edgelist[ind1][1]= Edgelist[ind2][1];
            Edgelist[ind2][1] = Temp1;
            cc= cc+1;
            //printf("%d\t", cc);

        }

    }

}

