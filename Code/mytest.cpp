//
//  main.cpp
//  test
//
//  Created by drinking on 1/2/16.
//  Copyright © 2016 drinking. All rights reserved.
//

#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <string>
#include <math.h>
#include <fstream>

using namespace std;
using namespace cv;

int PersonNumber,SampleNumber;
Mat EigenBase,NewCoord;
string ModelFolder = "Model";
string inputImgName;
string trainSetPath = "trainSet";
int width,height;

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

void GetInput()
{
    ifstream fin(ModelFolder+"/ModelData");
    fin >> PersonNumber >> SampleNumber;
    fin >> width >> height;
    CvMat *m = (CvMat*)cvLoad((ModelFolder+"/EigenBaseMat.xml").c_str());
    EigenBase = Mat(m);
    m = (CvMat*)cvLoad((ModelFolder+"/NewCoordMat.xml").c_str());
    NewCoord = Mat(m);
}

void MatchFace()
{
    Mat inputImg;
    Mat resultImg;
	vector<Mat> bgr;
	uchar** InputData = (uchar**)malloc(height*sizeof(double*));
	char number[20],name[20];
	inputImg = imread(inputImgName);
	resize(inputImg, inputImg, Size(width,height));
	split(inputImg, bgr);
	equalizeHist(bgr[0], inputImg);
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
	Mat distanceset = Mat(1,PersonNumber*SampleNumber,CV_64F);
	double* p = distanceset.row(0).ptr<double>();
	double* ptr;
	double* vecp = newvec.row(0).ptr<double>();
	for(int i=0;i<distanceset.cols;i++)
	{
		ptr = NewCoord.row(i).ptr<double>();
		p[i] = 0;
		for(int j=0; j<NewCoord.cols;j++)
			p[i]+= (ptr[j]-vecp[j])*(ptr[j]-vecp[j]);
		p[i] = sqrt(p[i]);
	}
	double min=100000000;
	int chosen=-1;
	for(int i=0;i<PersonNumber*SampleNumber;i++)
		if(min>p[i])
		{
			min = p[i];
			chosen = i;
		}
	sprintf(number, "/%d",chosen/SampleNumber+1);
	sprintf(name, "/%d.jpg",chosen%SampleNumber+1);
	char text[10];
	sprintf(text,"%d",chosen/SampleNumber+1);
	putText(inputImg, string(text), Point(10,10), 1, 1, Scalar(255,0,0));
	imshow("Input", inputImg);
	resultImg = imread(trainSetPath+number+name);
	imshow("Result", resultImg);
	waitKey();
}

int main(int argc, const char * argv[]) {
	if(argc<3) return -1;
	inputImgName = string(argv[1]);
	ModelFolder = string(argv[2]);
	if(argc == 4) trainSetPath = string(argv[3]);
    namedWindow("Result");
    namedWindow("Input");
    GetInput();
    MatchFace();
    return 0;
}
