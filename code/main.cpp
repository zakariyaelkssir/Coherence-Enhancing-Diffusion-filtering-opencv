#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/photo/photo.hpp"
#include "math.h"
#include "opencv2/xphoto.hpp"
#include <stdlib.h>
#include <unistd.h>
#include <iostream>


using namespace cv;

cv::Mat nonnegativitydiscretization(cv::Mat image,cv::Mat Dxx,cv::Mat Dyy,cv::Mat Dxy,float step){
	cv::Mat px(cv::Size(1,image.rows), CV_32F, cv::Scalar(0)); 
	cv::Mat py(cv::Size(1,image.cols), CV_32F, cv::Scalar(0)); 
	cv::Mat ny(cv::Size(1,image.rows), CV_32F, cv::Scalar(0)); 
	cv::Mat nx(cv::Size(1,image.cols), CV_32F, cv::Scalar(0)); 
	cv::Mat wbR1(image.size(), CV_32F, cv::Scalar(0)); 
	cv::Mat wbL3(image.size(), CV_32F, cv::Scalar(0)); 
	cv::Mat wtR7(image.size(), CV_32F, cv::Scalar(0)); 
	cv::Mat wtL9(image.size(), CV_32F, cv::Scalar(0)); 
	cv::Mat wtM2(image.size(), CV_32F, cv::Scalar(0)); 
	cv::Mat wmR4(image.size(), CV_32F, cv::Scalar(0)); 
	cv::Mat wmL6(image.size(), CV_32F, cv::Scalar(0));
	cv::Mat wmB8(image.size(), CV_32F, cv::Scalar(0)); 
	cv::Mat image_filter(image.size(), CV_32F, cv::Scalar(0)); 

	 nx.at<float>(0,0) = 0; 
	 ny.at<float>(0,0) = 0; 
	for (int i=0; i<image.rows; i++){
		px.at<float>(0,i) = i+1; 
	    ny.at<float>(0,i+1) = i; 
		}
		
	for(int i = 0;i<image.cols;i++){
		py.at<float>(0,i) = i+1;
		nx.at<float>(0,i+1) = i; 
	}
    px.at<float>(0,image.rows-1)=255;
	py.at<float>(0,image.cols-1)=255;
	// % Stencil Weights
	cv::Mat b,c,c2;
	c= cv::abs(Dxy)-Dxy;
	c2= cv::abs(Dxy)+Dxy;
	for(int i = 0 ;i<image.cols;i++){
		for(int j = 0;j<image.rows;j++){
			wbR1.at<float>(i,j) = 0.25*(abs(Dxy.at<float>(nx.at<float>(0,i),py.at<float>(0,j)))-Dxy.at<float>(nx.at<float>(0,i),py.at<float>(0,j))+c.at<float>(i,j));
			wbL3.at<float>(i,j) = 0.25*(abs(Dxy.at<float>(px.at<float>(0,i),py.at<float>(0,j)))+Dxy.at<float>(px.at<float>(0,i),py.at<float>(0,j))+c2.at<float>(i,j));
			wtR7.at<float>(i,j) = 0.25*(abs(Dxy.at<float>(nx.at<float>(0,i),py.at<float>(0,j)))+Dxy.at<float>(nx.at<float>(0,i),py.at<float>(0,j))+c2.at<float>(i,j));
			wtL9.at<float>(i,j) = 0.25*(abs(Dxy.at<float>(px.at<float>(0,i),py.at<float>(0,j)))-Dxy.at<float>(px.at<float>(0,i),py.at<float>(0,j))+c.at<float>(i,j));
			wtM2.at<float>(i,j) = 0.5*((Dyy.at<float>(i,py.at<float>(0,j))+Dyy.at<float>(i,j))-(abs(Dxy.at<float>(i,py.at<float>(0,j)))+abs(Dxy.at<float>(i,j))));
			wmR4.at<float>(i,j) = 0.5*((Dxx.at<float>(nx.at<float>(0,i),j)+Dxx.at<float>(i,j))-(abs(Dxy.at<float>(nx.at<float>(0,i),j))+abs(Dxy.at<float>(i,j))));
			wmL6.at<float>(i,j) = 0.5*((Dxx.at<float>(px.at<float>(0,i),j)+Dxx.at<float>(i,j))-(abs(Dxy.at<float>(px.at<float>(0,i),j))+abs(Dxy.at<float>(i,j))));
			wmB8.at<float>(i,j) = 0.5*((Dyy.at<float>(i,ny.at<float>(0,j))+Dyy.at<float>(i,j))-(abs(Dxy.at<float>(i,ny.at<float>(0,j)))+abs(Dxy.at<float>(i,j))));

			//abs(Dxy.at<float>(nx.at<float>(0,i),py.at<float>(0,j)));
			//std::cout<<nx.at<int>(i,j);
			image_filter.at<float>(i,j)=image.at<float>(i,j)+step*((wbR1.at<float>(i,j)*(image.at<float>(nx.at<float>(0,i),py.at<float>(0,j))-image.at<float>(i,j)))+(wtM2.at<float>(i,j)*(image.at<float>(i,py.at<float>(0,j))-image.at<float>(i,j)))+(wbL3.at<float>(i,j)*(image.at<float>(px.at<float>(0,i),py.at<float>(0,j))-image.at<float>(i,j)))+(wmR4.at<float>(i,j)*(image.at<float>(nx.at<float>(0,i),j)-image.at<float>(i,j)))+(wmL6.at<float>(i,j)*(image.at<float>(px.at<float>(0,i),j)-image.at<float>(i,j)))+(wtR7.at<float>(i,j)*(image.at<float>(nx.at<float>(0,i),ny.at<float>(0,j))-image.at<float>(i,j)))+(wmB8.at<float>(i,j)*(image.at<float>(i,ny.at<float>(0,j))-image.at<float>(i,j)))+(wtL9.at<float>(i,j)*(image.at<float>(px.at<float>(0,i),ny.at<float>(0,j))-image.at<float>(i,j))));
		}
	}
	//wbR1 = (0.25)*(cv::abs(Dxy)-Dxy);
	//image_filter.convertTo(image_filter,CV_8U);
//	cv::imshow("wmR4",image_filter);
	//cv::waitKey();
	return image_filter;
	}
static Mat gradientX(Mat & mat, float spacing) {
    Mat grad = Mat::zeros(mat.rows,mat.cols,CV_32F);

    /*  last row */
    int maxCols = mat.cols;
    int maxRows = mat.rows;

    /* get gradients in each border */
    /* first row */
    Mat col = (-mat.col(0) + mat.col(1))/(float)spacing;
    col.copyTo(grad(Rect(0,0,1,maxRows)));

    col = (-mat.col(maxCols-2) + mat.col(maxCols-1))/(float)spacing;
    col.copyTo(grad(Rect(maxCols-1,0,1,maxRows)));

    /* centered elements */
    Mat centeredMat = mat(Rect(0,0,maxCols-2,maxRows));
    Mat offsetMat = mat(Rect(2,0,maxCols-2,maxRows));
    Mat resultCenteredMat = (-centeredMat + offsetMat)/(((float)spacing)*2.0);

    resultCenteredMat.copyTo(grad(Rect(1,0,maxCols-2, maxRows)));
    return grad;
}


static Mat gradientY(Mat & mat, float spacing) {
    Mat grad = Mat::zeros(mat.rows,mat.cols,CV_32F);

    /*  last row */
    const int maxCols = mat.cols;
    const int maxRows = mat.rows;

    /* get gradients in each border */
    /* first row */
    Mat row = (-mat.row(0) + mat.row(1))/(float)spacing;
    row.copyTo(grad(Rect(0,0,maxCols,1)));

    row = (-mat.row(maxRows-2) + mat.row(maxRows-1))/(float)spacing;
    row.copyTo(grad(Rect(0,maxRows-1,maxCols,1)));

    /* centered elements */
    Mat centeredMat = mat(Rect(0,0,maxCols,maxRows-2));
    Mat offsetMat = mat(Rect(0,2,maxCols,maxRows-2));
    Mat resultCenteredMat = (-centeredMat + offsetMat)/(((float)spacing)*2.0);

    resultCenteredMat.copyTo(grad(Rect(0,1,maxCols, maxRows-2)));
    return grad;
}

int main(int argc, char** argv)
{
cv::Mat input = cv::imread(argv[1]); // change this line to load your actual input file

	// Make sure the input image is valid
	if (!input.data) {
		std::cerr << "The provided input image is invalid. Please check it again. " << std::endl;
		exit(1);
	}
while (input.rows > 1000 || input.cols > 1000) {
		const float fact = 0.6;
		cv::resize(input, input, cv::Size(), fact, fact, cv::INTER_CUBIC);
	}
	cv::Mat ori;
	input.copyTo(ori);
	float myalpha = 0.001,sigma=0.7;
	unsigned T = 15,rho = 4;
	int C= 1;
	float stepT=0.15;
	// %% 1 gaussian K_sigma
	float t=0.f;
	cv::Mat output;
		if (input.channels() != 1) {
		cv::cvtColor(input, input, CV_RGB2GRAY);
	}
    cv::Mat grady ;
	cv::Mat gradx ;
    
	input.convertTo(input,CV_32F);
	while(t<(T-0.001)){
	t = t + stepT;
	int limitXSize = ceil(2*sigma)-(-ceil(2*sigma))+1;
	cv::Mat limitX = cv::Mat::zeros(1, limitXSize, CV_32F);
	cv::Mat kSigma = cv::Mat::zeros(1, limitXSize, CV_32F);
	cv::Mat limitXX;
	for(int i = -ceil(2*sigma),j=0; i < ceil(2*sigma)+1; i++,j++){
    limitX.at<float>(j) = i;
	kSigma.at<float>(j) = cv::exp(-(i*i)/(2*pow(sigma,2)));
	}
	float s = cv::sum(kSigma)[0];
	for(int i = -ceil(2*sigma),j=0; i < ceil(2*sigma)+1; i++,j++){
	kSigma.at<float>(j)=kSigma.at<float>(j)/s;
		}	
			// Check whether the input image is grayscale.
	// If not, convert it to grayscale.

	
	 Point anchor(-1,-1);
     float delta = 0.0;

	 filter2D(input, output, CV_32F , kSigma.t(),anchor, 0, BORDER_REPLICATE );
	 filter2D(output, output, CV_32F , kSigma,anchor, 0, BORDER_REPLICATE );
	 //Imgproc.filter2D(Framesarray[i], TempmGray1, CvType.CV_32FC1, kernel, anchor, 0, Imgproc.BORDER_REPLICATE);
		//output.convertTo(output,CV_32F);

    
	grady = gradientY(output,1);
	gradx = gradientX(output,1);
        //

	/*// Run the enhancement algorithm
	FPEnhancement fpEnhancement;
	cv::Mat enhancedImage = fpEnhancement.run(output);
	std::cout << getImageType(enhancedImage.type()) << std::endl;
*/
	//%% 3 gaussian K_rho
	int limitXSizeJ = ceil(3*rho)-(-ceil(3*rho))+1;
	cv::Mat kSigmaJ = cv::Mat::zeros(1, limitXSizeJ, CV_32F);
    
	for(int i = -ceil(3*rho),j=0; i < ceil(3*rho)+1; i++,j++){
	kSigmaJ.at<float>(j) = cv::exp(-(i*i)/(2*pow(rho,2)));
	}
	s = cv::sum(kSigmaJ)[0];
    
	for(int i = -ceil(2*sigma),j=0; i < ceil(2*sigma)+1; i++,j++){
	kSigmaJ.at<float>(j)=kSigmaJ.at<float>(j)/s;
		}	
    
	cv::Mat Uxx,Jxx,Jyy,Jxy;
	cv::multiply(grady,grady,Jyy);
    cv::multiply(gradx,gradx,Jxx);
	cv::multiply(grady,gradx,Jxy);

	

	//calcul Jyy,Jxx,Jxy
	filter2D(Jyy,Jyy, CV_32F , kSigmaJ.t(),anchor, 0, BORDER_REPLICATE );
	filter2D(Jyy, Jyy, CV_32F , kSigmaJ,anchor, 0, BORDER_REPLICATE );

	filter2D(Jxx,Jxx, CV_32F , kSigmaJ.t(),anchor, 0, BORDER_REPLICATE );
	filter2D(Jxx, Jxx, CV_32F , kSigmaJ,anchor, 0, BORDER_REPLICATE );

	filter2D(Jxy,Jxy, CV_32F , kSigmaJ.t(),anchor, 0, BORDER_REPLICATE );
	filter2D(Jxy, Jxy, CV_32F , kSigmaJ,anchor, 0, BORDER_REPLICATE );

	
	
	//%% Principal axis transformation
    //% Eigenvectors of J, v1 and v2
	cv::Mat v2x,v2y,evec,eval,lamda1,lamda2,v1x,v1y,di,Dxx,Dyy,Dxy;
	//v2x.zeros(im2.size(),CV_32F);
    //v2y.zeros(im2.size(),CV_32F);
	input.convertTo(v2x,CV_32F);
	input.convertTo(v2y,CV_32F);
	input.convertTo(lamda1,CV_32F);
	input.convertTo(lamda2,CV_32F);
	v2x.zeros(input.size(),CV_32F);
    v2y.zeros(input.size(),CV_32F);
	lamda1.zeros(input.size(),CV_32F);
    lamda2.zeros(input.size(),CV_32F);
    
	for(int i =0 ; i<input.cols;i++){
		for(int j=0;j<input.rows;j++){
				cv::Mat pixel = cv::Mat::zeros(2,2, CV_32F);
	            pixel.at<float>(0,0)=Jxx.at<float>(i,j);
				pixel.at<float>(0,1)=Jxy.at<float>(i,j);
				pixel.at<float>(1,0)=Jxy.at<float>(i,j);
				pixel.at<float>(1,1)=Jyy.at<float>(i,j);
				cv::eigen(pixel,evec,eval);
				v2x.at<float>(i,j) = eval.at<float>(1,0);
				v2y.at<float>(i,j) = eval.at<float>(1,1);
				lamda1.at<float>(i,j) = evec.at<float>(0,1);
				lamda2.at<float>(i,j) = evec.at<float>(0,0);

				if(pow(v2x.at<float>(i,j),2)+pow(v2y.at<float>(i,j),2)!=0){

					v2x.at<float>(i,j)/=sqrt(pow(v2x.at<float>(i,j),2)+pow(v2y.at<float>(i,j),2));
					v2y.at<float>(i,j)/=sqrt(pow(v2x.at<float>(i,j),2)+pow(v2y.at<float>(i,j),2));

				}
		}
}

	v1x= -v2y;
	v1y=-v2x;


	
	di=(lamda1-lamda2);
	
	for(int i =0 ; i<input.cols;i++)
		for(int j=0;j<input.rows;j++){
			lamda1.at<float>(i,j)= myalpha +(1-myalpha)*exp(-1/(pow(di.at<float>(i,j),2)));
		}
	
	

	//Dxx = v1x*v1x;
	cv::Mat Dxx1,Dxx2,Dyy1,Dyy2,Dxy1,Dxy2;
	cv::multiply(v1x,v1x,Dxx1);
	cv::multiply(Dxx1,lamda1,Dxx1);
	cv::multiply(v1y,v1y,Dxx2);
	cv::multiply(myalpha,Dxx2,Dxx2);
     Dxx = Dxx1+Dxx2;

	cv::multiply(v1y,v1y,Dyy1);
	cv::multiply(lamda1,Dyy1,Dyy1);

	cv::multiply(v2y,v2y,Dyy2);
	cv::multiply(myalpha,Dyy2,Dyy2);

	cv::multiply(v1x,v1y,Dxy1);
	cv::multiply(lamda1,Dxy1,Dxy1);

	cv::multiply(v1y,v1y,Dxy2);
	cv::multiply(myalpha,Dxy2,Dxy2);

	
	Dyy = Dyy1+Dyy2;
	Dxy = Dxy1+Dxy2;

    
	//Dxx.convertTo(Dxx,CV_8U);
	//cv::imshow("lambda",Dxy);
	//cv::waitKey();
	





	input=nonnegativitydiscretization(input,Dxx,Dyy,Dxy,0.15);

	};

    	//Orientation of smooth gradient
	cv::Mat img_Orient(output.size(), CV_32F, cv::Scalar(0)); 
	int calc= 0;
	for(unsigned i=0 ; i<output.cols;i++){
		for(unsigned j =0;j<output.rows;j++){
			img_Orient.at<float>(i,j)=std::atan2(gradx.at<float>(i,j),grady.at<float>(i,j));
		}
	}

	input.convertTo(input,CV_8U);
	cv::imshow("Coherence Enhancing Diffusion Filtering :",input);
    double min;
    double max;
    cv::minMaxIdx(img_Orient, &min, &max);
    cv::Mat adjMap;
    // Histogram Equalization
    float scale = 255 / (max-min);
    img_Orient.convertTo(adjMap,CV_8U, scale, -min*scale);
    cv::Mat resultMap;
    applyColorMap(adjMap, resultMap, cv::COLORMAP_AUTUMN);
    cvtColor(resultMap, resultMap, CV_RGB2GRAY);
    cv::imshow("Out", resultMap);
    cv::imshow("Orientation of smooth gradient :",img_Orient);
	cv::waitKey();
	std::cout << "Press any key to continue... " << std::endl;
	cv::waitKey();
	cv::String locat= "../images/";
	imwrite(locat + argv[2], input);  
	return 0;
}
