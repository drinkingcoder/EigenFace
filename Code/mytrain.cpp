//
//  main.cpp
//  Lab3
//
//  Created by drinking on 12/26/15.
//  Copyright © 2015 drinking. All rights reserved.
//

#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <string>
#include <math.h>
#include <fstream>

#define IMAGENUM 54
#define SAMPLENUM 7

using namespace std;
using namespace cv;

Mat dataset[IMAGENUM*SAMPLENUM];
Mat meanMat;
Mat EigenBase;
Mat originalcoord;
uchar** DataSet[IMAGENUM*SAMPLENUM];
double* DistanceSet[IMAGENUM*SAMPLENUM];
Mat distanceset;
double** MeanMat;
double EnergyPropotion = 0.95;
Mat newcoordinate;
int width,height,length;

string inputPath = "trainSet/";
string MeanFaceName = "MeanFace.jpg";
string ModelFolder = "Model/";

void ToUnit(Mat& m)
{
    double **p = (double**)malloc(sizeof(double*)*m.rows);
    for(int i=0;i<m.rows;i++)
        p[i] = m.row(i).ptr<double>();
    for(int i=0;i<m.cols;i++)
    {
        double tmp = 0;
        for(int j=0; j<m.rows;j++)
            tmp+= p[j][i]*p[j][i];
        tmp = sqrt(tmp);
        for(int j=0; j<m.rows;j++)
            p[j][i] = p[j][i] / tmp;
    }
    free(p);
}

void PrintMatInfo(string s,Mat& m)
{
    cout << s << endl;
    cout << "\t" << "rows: " << m.rows << endl;
    cout << "\t" << "cols: " << m.cols << endl;
}

void InputFaces()
{
    char number[10],name[20];
    char eye[20];
    FILE* fp;
    number[0] = number[1] = number[2] = '\0';
    vector<Mat> bgr;
    int lx,ly,rx,ry,x,y;
    for(int i=0;i<IMAGENUM;i++)
    {
        sprintf(number, "%d",i+1);
        for(int j=0;j<SAMPLENUM;j++)
        {
            sprintf(name, "/%d.jpg",j+1);
            dataset[i*SAMPLENUM+j] = imread(inputPath+number+name);
            split(dataset[i*SAMPLENUM+j],bgr);
            equalizeHist(bgr[0], dataset[i*SAMPLENUM+j]);
     //       dataset[i*SAMPLENUM+j] = bgr[0];
        }
    }
    width = dataset[0].cols;
    height = dataset[0].rows;
    for(int i=0;i<IMAGENUM;i++)
        for(int j=0;j<SAMPLENUM;j++)
        {
            resize(dataset[i*SAMPLENUM+j], dataset[i*SAMPLENUM+j], Size(width,height));
            DataSet[i*SAMPLENUM+j] = (uchar**)malloc(dataset[i*SAMPLENUM+j].rows*sizeof(double*));
            for(int k=0;k<dataset[i*SAMPLENUM+j].rows;k++)
                DataSet[i*SAMPLENUM+j][k] = dataset[i*SAMPLENUM+j].row(k).ptr<uchar>();
        }
    originalcoord = Mat(IMAGENUM*SAMPLENUM,width*height,CV_64F);
    double** OriginalCoord = (double**)malloc(width*height*sizeof(double*));
    for(int k=0;k<IMAGENUM*SAMPLENUM;k++)
    {
        OriginalCoord[k] = originalcoord.row(k).ptr<double>();
        for(int i=0;i<height;i++)
            for(int j=0;j<width;j++)
                OriginalCoord[k][i*width+j] = DataSet[k][i][j];
    }
    transpose(originalcoord, originalcoord);
    ToUnit(originalcoord);
}

void GetMeanFace()
{
    meanMat = Mat(dataset[0].size(),CV_64F);
    MeanMat = (double**)malloc(meanMat.rows*sizeof(double*));
    width = dataset[0].cols;
    height = dataset[0].rows;
    for(int i=0;i<meanMat.rows;i++)
    {
        MeanMat[i] = meanMat.row(i).ptr<double>();
        memset(MeanMat[i],0,sizeof(double)*height);
    }
    for(int k=0;k<IMAGENUM*SAMPLENUM;k++)
        for(int i=0;i<height;i++)
            for(int j=0;j<width;j++)
                MeanMat[i][j] += DataSet[k][i][j];
    Mat showImage = Mat(height,width,CV_8U);
    uchar** ShowImage = (uchar**)malloc(sizeof(uchar*)*height);
    for(int i=0;i<height;i++)
        ShowImage[i] = showImage.row(i).ptr<uchar>();
    for (int i=0; i<height; i++)
        for(int j=0;j<width;j++)
        {
            MeanMat[i][j]/=IMAGENUM*SAMPLENUM;
            ShowImage[i][j] = uchar(MeanMat[i][j]);
        }
    imshow("images", showImage);
    imwrite(ModelFolder+MeanFaceName, showImage);
}

void GetEigenBase()
{
    double sum = 0;
    distanceset = Mat(IMAGENUM*SAMPLENUM, height*width, CV_64F);
    for(int k=0;k<IMAGENUM*SAMPLENUM;k++)
    {
        DistanceSet[k] = distanceset.row(k).ptr<double>();
        for(int i=0;i<height;i++)
            for(int j=0;j<width;j++)
                DistanceSet[k][i*width+j] = DataSet[k][i][j] - MeanMat[i][j];
    }
    Mat res,left,right;
    SVD::compute(distanceset, res, left, right);
    double* p = res.col(0).ptr<double>();
    for(int i=0;i<IMAGENUM+SAMPLENUM;i++)
	{
		p[i]*=p[i];
        sum+= p[i];
	}
    double d = 0;
    for(int i=0;i<IMAGENUM+SAMPLENUM;i++)
    {
        d+=p[i];
        if(d/sum>=EnergyPropotion)
        {
            length = i+1;
            break;
        }
    }
    EigenBase = right.rowRange(0, length);
    cout << "length = " << length << endl;
}

void ShowEignFace()
{
    Mat show;
    show = Mat(meanMat.size(),CV_8U);
    uchar** img = (uchar**)malloc(sizeof(uchar*)*height);
    double* eigenvector;
    double min,max,gap;
    for(int i=0;i<height;i++)
        img[i] = show.row(i).ptr<uchar>();
  //  transpose(EigenBase, EigenBase);
    PrintMatInfo("EigenBase", EigenBase);
    char eigenfacename[20];
    for(int k=0;k<10;k++)
    {
        min = 1000000;
        max = -1;
        eigenvector = EigenBase.row(k).ptr<double>();
        for(int i=0;i<height*width;i++)
        {
            if(min>eigenvector[i]) min = eigenvector[i];
            if(max<eigenvector[i]) max = eigenvector[i];
        }
        gap = max - min;
        for(int i=0;i<height;i++)
            for(int j=0;j<width;j++)
                img[i][j] = (uchar)((eigenvector[i*width+j]-min)/gap*255);
        sprintf(eigenfacename, "EigenFace %d.jpg",k);
        imwrite(ModelFolder+eigenfacename, show);
		imshow("images",show);
		waitKey();
    }
}

void CalcNewCoordinate()
{
    newcoordinate = EigenBase*originalcoord;
    transpose(newcoordinate, newcoordinate);
}

void MatchFace()
{
    string inputImgName;
    Mat inputImg;
    vector<Mat> bgr;
    uchar** InputData = (uchar**)malloc(height*sizeof(double*));
    while(1)
    {
        cout << "Input Face Img:" << endl;
        cin >> inputImgName;
        if(inputImgName.compare("exit") == 0) break;
        inputImg = imread(inputPath+inputImgName);
        resize(inputImg, inputImg, Size(width,height));
        split(inputImg, bgr);
        equalizeHist(bgr[0], inputImg);
        imshow("images", inputImg);
        for(int i=0;i<height;i++)
            InputData[i] = inputImg.row(i).ptr<uchar>();
        Mat originalvec = Mat(1,width*height,CV_64F);
        double* OriginalVec = originalvec.row(0).ptr<double>();
        for(int i=0;i<height;i++)
            for(int j=0;j<width;j++)
                OriginalVec[i*width+j] = InputData[i][j];
        transpose(originalvec, originalvec);
        ToUnit(originalvec);
        Mat newvec = EigenBase*originalvec;
        transpose(newvec, newvec);
        Mat distanceset = Mat(1,SAMPLENUM*IMAGENUM,CV_64F);
        double* p = distanceset.row(0).ptr<double>();
        double* ptr;
        double* vecp = newvec.row(0).ptr<double>();
        for(int i=0;i<distanceset.cols;i++)
        {
            ptr = newcoordinate.row(i).ptr<double>();
            p[i] = 0;
            for(int j=0; j<newcoordinate.cols;j++)
                p[i]+= (ptr[j]-vecp[j])*(ptr[j]-vecp[j]);
            p[i] = sqrt(p[i]);
        }
        double min=100000000;
        int chosen=-1;
        for(int i=0;i<IMAGENUM*SAMPLENUM;i++)
            if(min>p[i])
            {
                min = p[i];
                chosen = i;
            }
        cout << "chosen = " << chosen << endl;
        cout << "min = " << min << endl;
        imshow("images", dataset[chosen]);
        waitKey();
    }
}

void ExportData()
{
    ofstream fout(ModelFolder+"ModelData");
    fout << IMAGENUM << ' ' << SAMPLENUM << endl;
    fout << width << ' ' << height << endl;
    CvMat m = EigenBase;
    cvSave((ModelFolder+"EigenBaseMat.xml").c_str(),&m);
    m = newcoordinate;
    cvSave((ModelFolder+"NewCoordMat.xml").c_str(),&m);
}

int main(int argc, const char * argv[]) {
	if(argc<3) return -1;
	EnergyPropotion = atof(argv[1]);
	ModelFolder = string(argv[2]);
	ModelFolder = ModelFolder+'/';
	if(argc>=4)
	{
		inputPath = string(argv[3]);
		inputPath = inputPath+'/';
	}
    namedWindow("images");
    cout << "Inputing Faces..." << endl;
    InputFaces();
    cout << "Get MeanFace.." << endl;
    GetMeanFace();
    cout << "Get EigenFace.."<< endl;
    GetEigenBase();
    cout << "ShowEigenFaces"   << endl;
    ShowEignFace();
    cout << "CalcNewCoordinate" << endl;
    CalcNewCoordinate();
    cout << "ExportData" << endl;
  //  MatchFace();
    ExportData();
    cout << "Finished" << endl;
    destroyWindow("images");
   // Prepare();
    return 0;
}
