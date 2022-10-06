
#include <iostream>
#include <string>
#include <hlib-io.hh>
#include <boost/format.hpp>
#include <tbb/parallel_for.h>
#include <hlib.hh>
#include <hpro/cluster/TClusterBasisBuilder.hh>
#include <hpro/algebra/mul_vec.hh>
#include <hpro/vector/TScalarVector.hh>

using namespace HLIB;
using namespace std;
using  real_t    = HLIB::real;
using namespace BLAS;

void log(std::string S){
  std::cout << S << std::endl;
}
void log(const std::vector< idx_t > & S){
  std::string sep = "";
  for(auto s : S)
    {
      std::cout << sep << s;
      sep= ",";
    }
    std::cout << std::endl;
}



class TLogCoeffFn : public TCoeffFn< real_t > {
private:
  
    std::unique_ptr<TMatrix> _N;

public:
    TLogCoeffFn ( const double h)
  

    {
                  _N = read_matrix( "examples/weightmat_resnet_1000_512.mat" );
    }
     virtual void eval  ( const std::vector< idx_t > &  rowidxs,
                       const std::vector< idx_t > &  colidxs,
                       real_t *                      matrix ) const

  {

        const size_t  n = rowidxs.size();
        const size_t  m = colidxs.size();
        const int rows=_N->rows();
        const int cols=_N->cols();
        for ( size_t  j = 0; j < m; ++j )
        {
            const int  idx1 = colidxs[ j ];

          for ( size_t  i = 0; i < n; ++i )
            {
                const int  idx0 = rowidxs[ i ];
                double     value = _N ->entry(idx0,idx1); 
                matrix[ j*n + i ] =real_t(value);
                
            }
        }

    }
    using TCoeffFn< real_t >::eval;

} ;
// Set Coordinaten
std::unique_ptr<TClusterTree> build_ct(size_t n ){
  const double             h = 1;
  std::vector<double*> vertices(n);
  std::vector<double*>bbmin(n,NULL);
  std::vector<double*>bbmax(n,NULL);
  for (int i=0;i <n;i++)
  {
      vertices[i]    = new double[1];
      vertices[i][0] = h * double(i) + ( h / 2.0 );
      bbmin[i]       = new double[1];
      bbmin[i][0]    = h * double(i);
      bbmax[i]       = new double[1];
      bbmax[i][0]    = h * double(i+1);
        
  }
  auto  coord = std::make_unique< TCoordinate >( vertices, 2, bbmin , bbmax); 
  //TCardBSPPartStrat  part_strat;
  TAutoBSPPartStrat    part_strat;
  TBSPCTBuilder        ct_builder(&part_strat);
  return ct_builder.build(coord.get());
}


int main ( int argc, char ** argv ) {


  try {
    INIT();
    int maxnumber=14;
    unique_ptr< TScalarVector > MemoryHMatrix =make_unique<TScalarVector>(maxnumber-1,0,real_valued);
    unique_ptr< TScalarVector > MemoryMatrix =make_unique<TScalarVector>(maxnumber-1,0,real_valued);
    unique_ptr< TScalarVector > timeMultWithHMatrix =make_unique<TScalarVector>(maxnumber-1,0,real_valued);
    unique_ptr< TScalarVector > timeMultWithMatrix =make_unique<TScalarVector>(maxnumber-1,0,real_valued);
    unique_ptr< TScalarVector > timeBuildHMAtrix =make_unique<TScalarVector>(maxnumber-1,0,real_valued);
    unique_ptr< TScalarVector > accHMatrix =make_unique<TScalarVector>(maxnumber-1,0,real_valued);
    

    for (int number=1;number<maxnumber;number++){
    CFG::set_verbosity( 3 );
    double accuracyiteration=0.1*number;

    unsigned int rank=10*(number-1);
    if(rank==0){
      rank=1;

    }
    accHMatrix ->set_entry(number-1,rank);

    cout<<"itartion number "<< number  << endl;   
    cout<<"Rank "<< rank  << endl;  

    // Cluster Tree
    const size_t             n = 512;   //number of columns
    const size_t             m = 1000;   //number of rows
    auto rowct =build_ct(m);
    auto colct=build_ct(n);

    // Block Cluster Tree
    TStdGeomAdmCond      adm( 2 );
    TBCBuilder  bct_builder;
    auto        bct = bct_builder.build( rowct.get(), colct.get(), & adm );
    TBlockCluster *  root = bct->root();


    // H-matrix builder
    TLogCoeffFn               log_coefffn( 1.0);
    TPermCoeffFn< real_t >    coefffn( & log_coefffn, rowct->perm_i2e(), colct->perm_i2e() );
    TSVDLRApx   < real_t >           aca(& coefffn );   //TSVDLRApx ,TRandSVDLRApx, TACA ,TRRQRLRApx 
    auto  acc = fixed_rank(rank);
    //TTruncAcc                 acc(accuracyiteration);
    TDenseMBuilder < real_t >  h_builder( & coefffn, & aca );
    TTimer                    timer( WALL_TIME );
    timer.start();
    auto                      H = h_builder.build( bct.get(), acc );
    timer.pause();

    timeBuildHMAtrix ->set_entry(number-1,timer.elapsed());

   //Matrix properties & save Matrix
    auto   M = read_matrix( "examples/weightmat_resnet_1000_512.mat" );
    std::cout << "    done in " << timer << std::endl;
    std::cout << "    size of matrix = " << Mem::to_string( M->byte_size() ) << std::endl;
    std::cout << "    size of H-matrix = " << Mem::to_string( H->byte_size() ) << std::endl;
    string str = "visualization/resnet/HmobileSVD" ;
    string mat = ".mat" ;
    auto s = std::to_string(rank);
    string filename =str+s+mat;
    std::cout<< filename<<std::endl;
    write_matrix( H.get(), filename );


   //Visualization
    TPSMatrixVis  mvis;
     mvis.svd(true).print( H.get(), "HMAtrixSVD" );  
      TPSBlockClusterVis   bc_vis;
      bc_vis.print( root, "HMat_bct" );
  
  // Vector - Matrix muliplication
  unique_ptr< TVector >  b, x, sol ;
  b   = H->row_vector();
  sol = H->col_vector();
  sol->fill( 1.0 );

  int numbermult=100000;      // Number of MAtrix - vector mulitpliaktions 100000
  if(number==1){
  for (int i=0;i< 1*numbermult;i++){
  M->mul_vec( 1.0, sol.get(), 0.0, b.get(), apply_normal );
  }
  }

/*
  timer.start();
  for (int i=0;i<numbermult;i++){
  M->mul_vec( 1.0, sol.get(), 0.0, b.get(), apply_normal );
  }
  timer.pause();
  auto timerNormal= timer;
  timeMultWithMatrix->set_entry(number-1,timer.elapsed());
  */
  MemoryHMatrix->set_entry(number-1,H->byte_size());
  MemoryMatrix->set_entry(number-1,M->byte_size());
  //std::cout << "   Time of " << numbermult << " matrix-vector multiplications with matrix " << timer << std::endl;


  
  timer.start();
  for (int i=0;i<numbermult;i++){
   H->mul_vec( 1.0, sol.get(), 0.0, b.get(), apply_normal );
  }
  timer.pause();
  
  timeMultWithHMatrix ->set_entry(number-1,timer.elapsed());

  std::cout << "   Time of " << numbermult << " matrix-vector multiplications with  H-matrix " << timer << std::endl;
  

  std::cout<<"Finish Iteration"<< std::endl;
  }
 // write_vector( MemoryMatrix.get(), "visualization/resnet/MemoryMAtrix.mat" );
  //write_vector( MemoryHMatrix.get(), "visualization/resnet/MemoryQRrank.mat" );
  //write_vector( timeMultWithMatrix.get(), "visualization/resnet/MultwithMatrixSVDrank.mat" );
  write_vector( timeMultWithHMatrix.get(), "visualization/resnet/MultwithHMatrixSVDrank.mat" );
  //write_vector( timeBuildHMAtrix.get(), "visualization/resnet/timeBuildHMatrixSVDrank.mat" );
  //write_vector( accHMatrix.get(), "visualization/resnet/HMatrixSVDrank.mat" );
  
    DONE();
  }// try
  catch ( Error & e )
  {
    std::cout << e.to_string() << std::endl;
  }// catch

  return 0;
}
