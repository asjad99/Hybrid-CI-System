#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <cstdio>

using namespace std;

  const int cTarget = 5;          // Number we want mlp to recognise
  const int cNumPatterns = 2000;   // Number of training patterns

  //---------------MLP var intialization-------------

  long  NumIts = 15;    // Max training iterations
  int NumIPs = 784;
  const int NumOPs = 1;
  const int NumTrnPats = 2000;
 // const int NumTstPats = 200;

  int NumHN = 2;
  int NumHN1 = 18;
  int NumHN2 =  12;

  //read learn_rate,momentum, objective ...
  float LrnRate = 0.6;
  float Mtm1 = 1.2;
  float Mtm2 = 0.4;
  float ObjErr = 0.0020;
  int Ordering = 0;

// mlp weights
float **w1,**w11,**w111;// 1st layer wts
float **w2,**w22,**w222;// 2nd layer wts
float **w3,**w33,**w333;// 3rd layer wts

 
//---------------MLP var intialization ends-------------


const int cDebug = 0;

enum Xover{eRandom,eUniform,eOnePoint,eTwoPoint};

const Xover  CrossoverType = eTwoPoint;
const double cCrossoverRate = 0.70;
const double cMutationRate = 0.001;
const int    cNumGens = 55;
const int    cPopSize = 100; // must be an even number
const int    cTournamentSize = 5;
const int    Seed = 1234;
const int    cTargetFitness=150;
const int    cIndividualLength=784;

float PercentageError = 0;
int ones_counter = 0;

void InitPop(int ***CrntPop,int ***NextPop,int **Fitness,int **BestMenber);
void FreeMem(int **CrntPop,int **NextPop,int *Fitness,int *BestMember);
int Tournament(int *Fitness,int TournamentSize);
int EvaluateFitness(int *Member,float **x,float **d);
void Crossover(int *P1,int *P2,int *C1,int *C2);
void Copy(int *P1,int *P2,int *C1,int *C2);
void Mutate(int *Member);
double Rand01();    // 0..1
int RandInt(int n); // 0..n-1

float TrainNet3(float **x,float **d,int NumIPs,int NumOPs,int NumPats);
float **Aloc2DAry(int m,int n);
void Free2DAry(float **Ary2D,int n);
long random_gen(long max);



//===========================================================

int main(int argc,char *argv[]){

  //---------------------------------------------------------------
  //read input data 
  //simultaneously read data into the two 2d arrays IPTrnData,OPTrnData of length 'NumTrnPats x NumIPs' and 'NumTrnPats x NumOPs' respectively
  //---------------------------------------------------------------
  int i,j;
  int Label,Pixel;
  ifstream finImage,finLabel;
  char Filename[100];

  float **IPTrnData = Aloc2DAry(NumTrnPats,NumIPs);
  float **OPTrnData = Aloc2DAry(NumTrnPats,NumOPs);

   sprintf(Filename,"Train%d.txt",cTarget);
  cout<<"Filename: "<<Filename<<endl<<endl;
  ifstream fin(Filename);
  if(!fin.good()){
    cout<<"Input file not found!\n";
    system("PAUSE");
    exit(1);
  }

  //Total NumTrnPats = 200 or 2000,5000 
  // initial NumIPs = 200 x 784
  // initial NumOPs = 200 x 1(single binary output for each image)
  //-----------------------------------------
  for(i=0;i<NumTrnPats;i++){
   for(j=0;j<NumIPs;j++){
     fin>>Pixel; //read input 28x28 image binary data
     IPTrnData[i][j] = Pixel;
   }
   fin >> Label;  //read output
   for(j=0;j<NumOPs;j++)
      OPTrnData[i][j] = Label;
  }

  fin.close();

  //for(i=0;i<1;i++){
  // for(j=0;j<NumIPs;j++){
  //    cout << IPTrnData[i][j];
  //  }
  //}

  //---------------------------------------------------------------
  //GA initialization 
  //---------------------------------------------------------------

  int **CrntPop, **NextPop; // the crnt & next population lives here
  int *Fitness, BestFitness=500, *BestMember; // fitness vars
  int BestFitness_pergeneration = 500;
 int TargetReached=false;

  InitPop(&CrntPop,&NextPop,&Fitness,&BestMember);

  ofstream myfile;
  myfile.open ("fitnessvsGen.csv");
  //myfile << "Train results.\n";
  myfile << "Generations,Fitness,PercentageError,ones_counter,\n";

  ofstream myfile2;
  myfile2.open ("fitnessvsPercentageError.csv");
  //myfile << "Train results.\n";
  myfile2 << "PercentageError,Fitness,\n";
 
  for(int Gen=0;Gen<cNumGens;Gen++){
    BestFitness_pergeneration = 500;

    for(i=0;i<cPopSize;i++){

      // Evaluate the fitness of pop members
      Fitness[i]=EvaluateFitness(CrntPop[i],IPTrnData,OPTrnData);
       myfile2 << PercentageError << "," << Fitness[i] <<"\n";
      if(BestFitness>Fitness[i]){ // save best member
        BestFitness=Fitness[i];
        for(int j=0;j<cIndividualLength;j++)BestMember[j]=CrntPop[i][j];
        if(Fitness[i]>=cTargetFitness){
          TargetReached=true;
          break;
        }
      }
      if (BestFitness_pergeneration>Fitness[i]){
         BestFitness_pergeneration = Fitness[i];

      }

    }

    myfile << Gen << "," << BestFitness_pergeneration << "," << PercentageError <<  "," << ones_counter <<"\n";
    
    if(TargetReached)break;

    // Produce the next population
    for(i=0;i<cPopSize;i+=2){
      int Parent1=Tournament(Fitness,cTournamentSize);
      int Parent2=Tournament(Fitness,cTournamentSize);
      if(cCrossoverRate>Rand01())
        Crossover(CrntPop[Parent1],CrntPop[Parent2],NextPop[i],NextPop[i+1]);
      else
        Copy(CrntPop[Parent1],CrntPop[Parent2],NextPop[i],NextPop[i+1]);
      if(cMutationRate<Rand01())Mutate(NextPop[i]);
      if(cMutationRate<Rand01())Mutate(NextPop[i+1]);
    }
    int **Tmp=CrntPop; CrntPop=NextPop; NextPop=Tmp;

    cout << "----------------------best fitness----------------------\n";
    cout << setw(3)<<Gen<<':'<<setw(5)<<BestFitness<<endl;
    cout << "\n--------------------------------------------------------";
    
  
  }
  //if(TargetReached) cout<<"Target fitness reached: "<<BestFitness<<"!\n";
  //else cout<<"Target fitness not reached: "<<BestFitness<<"!\n";
  myfile.close();
  myfile2.close();
  cout<<"Best Individual: ";
  for(i=0;i<cIndividualLength;i++)cout<<BestMember[i];cout<<endl;
  FreeMem(CrntPop,NextPop,Fitness,BestMember);
  char s[20];cin.getline(s,20);

    //DO this at the end of main
  Free2DAry(IPTrnData,NumTrnPats);
  Free2DAry(OPTrnData,NumTrnPats);

 

  return 0;


}
//===========================================================

void InitPop(int ***CrntPop,int ***NextPop,int **Fitness,int **BestMember){
  // declare two arrays of population size length. 
  //one holds current population, one is for next population
 
  int i, j;
  srand(Seed);
  int MaxIP = 30;

  *CrntPop = new int*[cPopSize];
  *NextPop = new int*[cPopSize];
  
  for(i=0;i<cPopSize;i++){
    (*CrntPop)[i] = new int[cIndividualLength]; 
    (*NextPop)[i] = new int[cIndividualLength];
  }
  *Fitness    = new int[cPopSize]; 
  *BestMember = new int[cIndividualLength]; 
  if(Fitness==NULL||BestMember==NULL)exit(1);
 
  for(i=0;i<cPopSize;i++){
    for(j=0;j<cIndividualLength;j++){
      (*CrntPop)[i][j] = 0;
    }
  }

  int filtered_columns = RandInt(30);

  for(i=0;i<cPopSize;i++){
    for(j=0;j<MaxIP;j++){
      int random_column = RandInt(cIndividualLength);
      (*CrntPop)[i][random_column] = 1;
    }
  }

} 

void FreeMem(int **CrntPop,int **NextPop,int *Fitness,int *BestMenber){
  for(int i=0;i<cPopSize;i++){
    delete[]CrntPop[i];
    delete[]NextPop[i];
  }
  delete CrntPop;
  delete NextPop;
  delete Fitness;
  delete BestMenber;
}
//===========================================================

int EvaluateFitness(int *Member,float **IPTrnData,float **OPTrnData){

 int TheFitness = 0;
 int colmn_num = 0;
 ones_counter = 0;

 int i,j;


  //count the total number of 1's occured in GA's member
 for(i=0;i<cIndividualLength;i++){
     if (Member[i]==1){
          ones_counter++;  //counter the number of 1's in the population member
      }
  }

  //store the column numbers where 1's occured


  int column_numbers[ones_counter-1];

  j=0;
  for(i=0;i<cIndividualLength;i++){
     
     if (Member[i]==1){
       
       column_numbers[j] = i;
       j++; //j should not exceed ones_counter
    }
  }
 
  // Perform filtering of IPTrnData based on column numbers 
  float **IPTrnData_filtered = Aloc2DAry(NumTrnPats,ones_counter);

  

  for(i=0;i<NumTrnPats;i++){
   for(j=0;j<ones_counter;j++){
     colmn_num= column_numbers[j];
  
     IPTrnData_filtered[i][j] = IPTrnData[i][colmn_num];
   }
 }

  int NumIPs = ones_counter;  //number of inputs to the mlp
  int NumOPs = 1;
  int NumTrnPats = 2000;
  int MinIP = 10;
  float MaxIP = 30;
  

  //Pass filtered data to train net
  PercentageError = 0;
  PercentageError = TrainNet3(IPTrnData_filtered,OPTrnData,NumIPs,NumOPs,NumTrnPats);  //3 layer mlp

  float c = 0.5; //0..1
  float FS = (NumIPs-MinIP) / (MaxIP-MinIP) * 100;

  int FE = PercentageError;
  //cout << PercentageError;

  TheFitness = c * FS + (1-c) * FE; // note: bigger is better

  cout <<"------------fitness--------\n";
  cout << TheFitness;
  cout <<"\n--------------------\n";


  return(TheFitness);
}
//================================================================

int Tournament(int *Fitness,int TournamentSize){
  int WinFit = -99999, Winner;
  for(int i=0;i<TournamentSize;i++){
    int j = RandInt(cPopSize);
    if(Fitness[j]>WinFit){
      WinFit = Fitness[j];
      Winner = j;
    }
  }
  return Winner;
}

void Crossover(int *P1,int *P2,int *C1,int *C2){
  int i, Left, Right;
  switch(CrossoverType){
    case eRandom: // swap random genes
      for(i=0;i<cIndividualLength;i++){
        if(RandInt(2)){
          C1[i]=P1[i]; C2[i]=P2[i];
        }else{
          C1[i]=P2[i]; C2[i]=P1[i];
        }
      }
      break;
    case eUniform: // swap odd/even genes
      for(i=0;i<cIndividualLength;i++){
        if(i%2){
          C1[i]=P1[i]; C2[i]=P2[i];
        }else{
          C1[i]=P2[i]; C2[i]=P1[i];
        }
      }
      break;
    case eOnePoint:  // perform 1 point x-over
      Left = RandInt(cIndividualLength);
      if(cDebug){
        printf("Cut points: 0 <= %d <= %d\n",Left,cIndividualLength-1);
      }
      for(i=0;i<=Left;i++){
        C1[i]=P1[i]; C2[i]=P2[i];
      }
      for(i=Left+1;i<cIndividualLength;i++){
        C1[i]=P2[i]; C2[i]=P1[i];
      }
      break;
    case eTwoPoint:  // perform 2 point x-over
      Left = RandInt(cIndividualLength -1);
      Right = Left+1+RandInt(cIndividualLength-Left-1);
      if(cDebug){
        printf("Cut points: 0 <= %d < %d <= %d\n",Left,Right,cIndividualLength-1);
      }
      for(i=0;i<=Left;i++){
        C1[i]=P1[i]; C2[i]=P2[i];
      }
      for(i=Left+1;i<=Right;i++){
        C1[i]=P2[i]; C2[i]=P1[i];
      }
      for(i=Right+1;i<cIndividualLength;i++){
        C1[i]=P1[i]; C2[i]=P2[i];
      }
      break;
    default:
      printf("Invalid crossover?\n");
      exit(1);
  }
}

void Mutate(int *Member){
  //About: Uses the NFIS Mutate Feature
  int Pa,Pi;
  int i,j;
  int inf_bit = 0;  
  int irBit = 0;
  int rf = 0;
  
  int ni=0;
  int sum =0;

  if (Pa <= Rand01()){
    // Adds the most informative feature among the set of features
    // that are not present in the current individual.

    for(i=0;i<cIndividualLength;i++){
     if (Member[i]==1){
          ni++; 
      }
    }
      inf_bit =  RandInt(cIndividualLength)- (ni/784);

     //set the irBit bit of the member to zero thus generating a new individual
      Member[inf_bit] = 1;

  }

  else if (Pi <= Rand01()){
      //Removes the most irrerelevent feature among the 
      //set of features already present in the current member
    for(i=0;i<cIndividualLength;i++){
     if (Member[i]==0){
          ni++; 
      }
    } 
    irBit = (ni - RandInt(cIndividualLength))/784;

      //set the irBit bit of the member to zero thus generating a new individual
      Member[irBit] = 0;
  }

  else  {

    //Remove the most redundant feature among the
    //set of features already present in the current member

     for(i=0;i<cIndividualLength;i++){
     if (Member[i]==0){
          sum++; 
      }
    } 
    irBit = (ni - (sum/784));

    //set the irBit bit of the member to zero thus generating a new individual
    Member[irBit] = 0;

   //set the rf bit of the member to zero thus generating a new individual
    Member[rf] = 0;
  }
}



void Copy(int *P1,int *P2,int *C1,int *C2){
  for(int i=0;i<cIndividualLength;i++){
    C1[i]=P1[i]; C2[i]=P2[i];
  }
}
//=================================================================

double Rand01(){ // 0..1
  return(rand()/(double)(RAND_MAX));
}

int RandInt(int n){ // 0..n-1
  return int( rand()/(double(RAND_MAX)+1) * n );
}

//-----------------------------------------------------------------

float TrainNet3(float **x,float **d,int NumIPs,int NumOPs,int NumPats ){

  cout << "------total patterns---------------\n";
  cout << NumIPs;
  cout << "---------------------";
      

// Trains 3 layer back propagation neural network
// x[][]=>input data, d[][]=>desired output data
   float PcntErr=0;

  float *h1 = new float[NumHN1]; // O/Ps of hidden layer
  float *h2 = new float[NumHN2]; // O/Ps of hidden layer 2
  
  float *y  = new float[NumOPs]; // O/P of Net
  
  float *ad1= new float[NumHN1]; // HN1 back prop errors
  float *ad2= new float[NumHN2]; // O/P back prop errors
  float *ad3= new float[NumOPs]; // O/P back prop errors

  float PatErr,MinErr,AveErr,MaxErr;  // Pattern errors
  int p,i,j;     // for loops indexes
  long ItCnt=0;  // Iteration counter
  long NumErr=0; // Error counter (added for spiral problem)

  ofstream myfile;
  myfile.open ("NNLearningratevsTrainingEpochs.csv");
  myfile << "Train results.\n";
  myfile << "IternationCount,MinErr,AverageErr,MaxError,PercentageError,PatternError,\n";
  

  cout<<"TrainNet3: IP:"<<NumIPs<<" H1:"<<NumHN1<< "H2:"<< NumHN2<< " OP:"<<NumOPs<<endl;

  // Allocate memory for weights
  w1   = Aloc2DAry(NumIPs,NumHN1);// 1st layer wts
  w11  = Aloc2DAry(NumIPs,NumHN1);
  w111 = Aloc2DAry(NumIPs,NumHN1);
  
  w2   = Aloc2DAry(NumHN1,NumHN2);// 2nd layer wts
  w22  = Aloc2DAry(NumHN1,NumHN2);
  w222 = Aloc2DAry(NumHN1,NumHN2);

  w3   = Aloc2DAry(NumHN2,NumOPs);// 3rd layer wts
  w33  = Aloc2DAry(NumHN2,NumOPs);
  w333 = Aloc2DAry(NumHN2,NumOPs);
    
    
  // Init wts between -0.5 and +0.5
  srand(time(0));
  for(i=0;i<NumIPs;i++)
    for(j=0;j<NumHN1;j++)
    w1[i][j]=w11[i][j]=w111[i][j]= float(rand())/RAND_MAX - 0.5;
  for(i=0;i<NumHN1;i++)
    for(j=0;j<NumHN2;j++)
      w2[i][j]=w22[i][j]=w222[i][j]= float(rand())/RAND_MAX - 0.5;
  for(i=0;i<NumHN2;i++)
    for(j=0;j<NumOPs;j++)
      w3[i][j]=w33[i][j]=w333[i][j]= float(rand())/RAND_MAX - 0.5;
     
     
    
  for(;;){// Main learning loop
    MinErr=3.4e38; AveErr=0; MaxErr=-3.4e38; NumErr=0;

    for(p=0;p<NumPats;p++){ // for each pattern...

      int rand_index = random_gen(NumPats-1);

      // Cal neural network output
     for(i=0;i<NumHN1;i++){ // Cal O/P of hidden layer 1
        float in=0;
        for(j=0;j<NumIPs;j++)
          in+=w1[j][i]*x[p][j];
        h1[i]=(float)(1.0/(1.0+exp(double(-in))));// Sigmoid fn
       // h1[i] = (float)tanh(in);
      }
      

      for(i=0;i<NumHN2;i++){  //Cal O/P of hidden layer 2
        float in=0;
        for(j=0;j<NumHN1;j++){
          in+=w2[j][i]*h1[j];
        }
        h2[i]=(float)(1.0/(1.0+exp(double(-in))));// Sigmoid fn
      }


      for(i=0;i<NumOPs;i++){ // Cal O/P of output layer

        float in=0;
        for(j=0;j<NumHN2;j++){
          in+=w3[j][i]*h2[j];
      }

      y[i]=(float)(1.0/(1.0+exp(double(-in))));// Sigmoid fn
      }


      // Cal error for this pattern
      PatErr=0.0;
      for(i=0;i<NumOPs;i++){
        float err=y[i]-d[p][i]; // actual-desired O/P
        if(err>0)PatErr+=err; else PatErr-=err;
        NumErr += ((y[i]<0.5&&d[p][i]>=0.5)||(y[i]>=0.5&&d[p][i]<0.5));//added for binary classification problem
     }


      if(PatErr<MinErr)MinErr=PatErr;
      if(PatErr>MaxErr)MaxErr=PatErr;
      AveErr+=PatErr;
      
      // Learn pattern with back propagation

      //error correction
      //-------------------------------------------------
       for(i=0;i<NumOPs;i++){ // Modify layer 3(output) wts
      // ad3[i]=(d[p][i]-y[i])*(1-tanh(y[i]))*(1+tanh(y[i]));
        ad3[i]=(d[p][i]-y[i])*y[i]*(1.0-y[i]);
      
      }

      for(i=0;i<NumHN2;i++){ // Modify layer 2 wts
        float err=0.0;
        for(j=0;j<NumOPs;j++)
          err+=ad3[j]*w3[i][j];
        ad2[i] = err*h2[i]*(1.0-h2[i]);
      }


      for(i=0;i<NumHN1;i++){ // Modify layer 1 wts
        float err=0.0;
        for(j=0;j<NumHN2;j++)
          err+=ad2[j]*w2[i][j];
        
        //ad1[i]=err*(1-tanh(h1[i]))*(1+tanh(h1[i]));
         ad1[i] = err *h1[i]*(1.0-h1[i]);

      }

     
      //------weights correction------------

      for(i=0;i<NumHN1;i++){ // Modify layer 1 wts
        
        for(j=0;j<NumIPs;j++){
          w1[j][i]+=LrnRate * x[p][j] *ad1[i]+
                    Mtm1*(w1[j][i]-w11[j][i])+
                    Mtm2*(w11[j][i]-w111[j][i]);
          w111[j][i]=w11[j][i];
          w11[j][i]=w1[j][i];
        }
      }

      for(i=0;i<NumHN2;i++){ // Modify layer 2 wts
        for(j=0;j<NumHN1;j++){
          w2[j][i]+=LrnRate*h1[j]*ad2[i]+
                    Mtm1*(w2[j][i]-w22[j][i])+
                    Mtm2*(w22[j][i]-w222[j][i]);
          w222[j][i]=w22[j][i];
          w22[j][i]=w2[j][i];
        }
      }



      for(i=0;i<NumOPs;i++){ // Modify layer 3(output) wts
        for(j=0;j<NumHN2;j++){
          w3[j][i]+=LrnRate*h2[j]*ad3[i]+
                    Mtm1*(w3[j][i]-w33[j][i])+
                    Mtm2*(w33[j][i]-w333[j][i]);
          w333[j][i]=w33[j][i];
          w33[j][i]=w3[j][i];
        }
      }
      //end weights correction
  
    }// end for each pattern
      
   ItCnt++;
    AveErr/= NumPats;
    PcntErr = NumErr/float(NumPats) * 100.0;
    cout.setf(ios::fixed|ios::showpoint);
    cout<<setprecision(6)<<setw(6)<<ItCnt<<":"<<setw(12)<<MinErr<<setw(12)<<AveErr<<setw(12)<<MaxErr<<setw(12)<<PcntErr <<endl;
    myfile << ItCnt <<"," << MinErr << "," << AveErr << "," << MaxErr  <<"," <<  PcntErr << "," << PatErr <<"\n";
 
   if((AveErr<=ObjErr)||(ItCnt==NumIts)) break;


  }// end main learning loop

 myfile.close();

  // Free memory
  delete h1; delete h2;delete y; 
  delete ad1; delete ad2;delete ad3;

  return PcntErr;
}

//----------------------------------------------
float **Aloc2DAry(int m,int n){
//Allocates memory for 2D array
  float **Ary2D = new float*[m];
  if(Ary2D==NULL){cout<<"No memory!\n";exit(1);}
  for(int i=0;i<m;i++){
   Ary2D[i] = new float[n];
   if(Ary2D[i]==NULL){cout<<"No memory!\n";exit(1);}
  }
  return Ary2D;
}


//----------------------------------------------
void Free2DAry(float **Ary2D,int n){
//Frees memory in 2D array
  for(int i=0;i<n;i++)
   delete [] Ary2D[i];
  delete [] Ary2D;
}


//----------------------------------------------
// Assumes 0 <= max <= RAND_MAX
// Returns in the half-open interval [0, max]
long random_gen(long max) {
  unsigned long
    // max <= RAND_MAX < ULONG_MAX, so this is okay.
    num_bins = (unsigned long) max + 1,
    num_rand = (unsigned long) RAND_MAX + 1,
    bin_size = num_rand / num_bins,
    defect   = num_rand % num_bins;

  long x;
  do {
   x = random();
  }
  // This is carefully written not to overflow
  while (num_rand - defect <= (unsigned long)x);

  // Truncated division is intentional
  return x/bin_size;
}


