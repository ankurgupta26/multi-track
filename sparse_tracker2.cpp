#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
//#include <time>
#include <iostream>
#define BIN_SIZE 8

using namespace cv;
using namespace std;

template<typename T>
	T pow2(T val)
	{
		return val * val;
	}

void evaluateFeatures(Mat ROI, Mat& features, Point2f p)
{
	//std::fill(hist,hist+SIZE,0);
	features = Mat::zeros(1, BIN_SIZE*BIN_SIZE*BIN_SIZE + 2, CV_64F);
	cv::Mat Lab_ROI;
	cv::Rect crop(0.15*ROI.cols, 0.15*ROI.rows, 0.7*ROI.cols, 0.7*ROI.rows);

	cv::Mat shrinkROI = ROI(crop);

	Lab_ROI = shrinkROI;
	cv::cvtColor(shrinkROI,Lab_ROI,CV_BGR2Lab);
	//resize(Lab_ROI, Lab_ROI, POP);
	uchar gap = 256/BIN_SIZE;

	float h_w = Lab_ROI.cols/2.0f;
	float h_h = Lab_ROI.rows/2.0f;
	float sum_location = 0;

	for (int i = 0; i<Lab_ROI.cols; ++i)
	{
		for (int j = 0; j < Lab_ROI.rows; ++j)
		{
			cv::Vec3b val = Lab_ROI.at<cv::Vec3b>(j,i);
			uchar L = val.val[0];
			uchar a = val.val[1];
			uchar b = val.val[2];
			int indexB = L / gap;
			int indexG = a / gap;
			int indexR = b / gap;
			//weight of this location 
			float location = pow2((i-h_w)/h_w) + pow2((j-h_h)/h_h);
			if (location<=1)
			{
				float weight = 0.75f*(1-location);
				features.at<double>(0, (indexG*BIN_SIZE+indexR)*BIN_SIZE+indexB) += weight;
				sum_location += weight;
			}
		}
	}
	int i;
	for (i = 0; i < BIN_SIZE*BIN_SIZE*BIN_SIZE; ++i)
	{
		features.at<double>(0, i) /= sum_location;
		features.at<double>(0, i) *= 10.8f;
	}
		//i = BIN_SIZE*BIN_SIZE*BIN_SIZE;
	//sum_location = (0.8*p.x + 0.8*p.y);
	features.at<double>(0, i) = 2.2f*p.x;// / sum_location;
	features.at<double>(0, i + 1) = 2.2f*p.y;// / sum_location;
	/*hist[i + 2] = 0.8*p.x / sum_location;
	hist[i + 3] = 0.8*p.y / sum_location;*/
}

bool notIndices(int index, vector<int> selected_indices)
{
	bool flag = true;
	for(int i = 0; i < selected_indices.size(); i++)
		if(selected_indices[i] == index)
		{
			flag = false;
			break;
		}
	return flag;
}

void OMP(Mat& dictionary, Mat& feature, Mat& coeff, int limit, Mat& residue)
{
	residue = feature.clone();
	coeff = Mat::zeros(1, dictionary.rows, CV_64F);
	Mat temp, phi;
	double inner_prod, max_inner_prod;
	int index = 0;
	vector<int> selected_indices;
	Mat P, inverse;// identity;
	if(dictionary.rows > 0)
	{
		for(int i = 0; i < limit; i++)
		{
			max_inner_prod = 0;
			inner_prod = 0;
			for(int j = 0; j < dictionary.rows; j++)
			{
				temp = (dictionary.row(j)).clone();
				inner_prod = temp.dot(feature);
				//cout << "Inner product of row " << j << " is "<< inner_prod << "\n";
				if(inner_prod > max_inner_prod && notIndices(j, selected_indices))
				{
					max_inner_prod = inner_prod;
					//cout << "Max Inner product of row " << j << " is "<< inner_prod << "\n";
					index = j;
				}
			}
			if(max_inner_prod > 0)
			{
				selected_indices.push_back(index);
				temp = dictionary.row(index).clone();
				//temp = temp*0.25;
				phi.push_back(temp);
				coeff.at<double>(0, index) = max_inner_prod;
				invert((phi * phi.t()), inverse, DECOMP_SVD);
				P = (phi.t() * (inverse * phi));
				Mat identity = Mat::eye(P.rows, P.rows, CV_64F);
				//cout << P << "\n";
				P = identity - P;
				//cout << P << "\n";
				residue = residue * P;
				//cout <<"Residue Mat is "<< residue << "\n";
			}
		}
	}
}

int main(int argc, char** argv)
{
	Mat im, dictionary, feature, coeff, residue, temp;
	vector<int> trackUpdateIndex;
	int num_tracks = 0, n_humans, track_label, conf_count, i;
	CascadeClassifier h_detector("hogcascade_pedestrians.xml");
	vector<Rect> humans;
	VideoCapture cap(argv[1]);
	bool flag = false;
	Rect r1(426, 109, 168, 212), r2(226, 109, 120, 212);
	int c1r1 = 0, c1r2 = 0, c2r1 = 0, c2r2 = 0;
	//cap >> im;
	//resize(im, im, Size(0, 0), 0.4, 0.4);
	//imwrite("test_img.jpg", im);
	while((char)waitKey(15) != 'a')
	{
		cap >> im;
		resize(im, im, Size(0, 0), 0.4, 0.4);
		h_detector.detectMultiScale(im, humans, 1.05, 4, 0, Size(70, 140), Size(200, 200));
		n_humans = humans.size();
		flag = false;
		for(i = 0; i < n_humans; i++)
		{
			humans[i].x = humans[i].x + humans[i].width/2 - humans[i].width*0.6/2;
			humans[i].y = humans[i].y + humans[i].height/2 - humans[i].height*0.75/2;
			humans[i].width *= 0.6;
			humans[i].height *= 0.75;
			evaluateFeatures(im(humans[i]).clone(), feature, Point2f((humans[i].x + humans[i].width / 2.0) / im.cols, (humans[i].y + humans[i].height / 2.0) / im.rows));
			feature /= norm(feature, NORM_L2);
			OMP(dictionary, feature, coeff, 4, residue);
			cout << "Residue norm " << norm(residue, NORM_L2) << "\n";
			if(norm(residue, NORM_L2) >= 0.3 && dictionary.rows < 8)
			{
				//if(dictionary.rows < 8)
				{
					//cout << "Added Data \n";
					//feature /= norm(feature, NORM_L2);
					Mat noise(1, feature.cols, CV_64F);
					//case of new data in dictionary
					dictionary.push_back(feature);
					randn(noise, 0, 0.5);
					temp = feature + noise;
					//cout << "Columns "<<noise.at<double>(0, 200) <<"\n";
					dictionary.push_back(feature);
					randn(noise, 0, 0.5);
					temp = feature + noise;
					dictionary.push_back(feature);
					randn(noise, 0, 0.5);
					temp = feature + noise;
					dictionary.push_back(feature);
					trackUpdateIndex.push_back(0);
				}
			}
			else
			{
				int j;
				double score, max_score = 0;
				int selection;
				//cout << "Number of tracks " << trackUpdateIndex.size() << "\n";
				for(j = 0; j < trackUpdateIndex.size(); j++)
				{
					conf_count = 0;
					for(int k = 0; k < 4; k++)
					{
						cout << "Coefficient "<< j << " " << coeff.at<double>(0, j*4 + k) <<"\n";
						if(coeff.at<double>(0, j*4 + k) > max_score)
						{
							max_score =  coeff.at<double>(0, j*4 + k);
							selection = j;
						}
					}
					//if(conf_count >= 1)
					//	break;
				}
				if(max_score > 0)
				{
					feature.copyTo(dictionary.row(selection*4 + trackUpdateIndex[selection]));
					//for(int k = 0; k < 4; k++)
					//	feature.copyTo(dictionary.row(k));
					trackUpdateIndex[selection] = (trackUpdateIndex[selection] + 1) % 4;
					if(selection == 0)// && !flag)
					{
						if((r1 & humans[i]).area() == humans[i].area())
							c1r1++;
						else if((r2 & humans[i]).area() == humans[i].area())
							c1r2++;
						rectangle(im, humans[i], Scalar(255, 0, 0));
						flag = true;
					}
					else if(selection == 1)// && flag)
					{
						if((r1 & humans[i]).area() == humans[i].area())
							c2r1++;
						else if((r2 & humans[i]).area() == humans[i].area())
							c2r2++;
						rectangle(im, humans[i], Scalar(0, 0, 255));
					}
				}
			}
		}
		//rectangle(im, r1, Scalar(0, 255, 0));
		//rectangle(im, r2, Scalar(0, 255, 0));
		imshow("Result", im);
	}
	cout << "c1r1 = " << c1r1 << "c1r2 = " << c1r2 << "c2r1 = "<< c2r1 << "c2r2 = " << c2r2 << "\n";
	cap.release();
	destroyAllWindows();
	return 0;
}