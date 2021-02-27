#include <iostream>
#include <memory>
#include <ctime>
#include <ratio>
#include <chrono>


#include <fstream>
#include <string>
#include <stdio.h>
#include <cmath>
#include <sstream>

using namespace std::chrono;
using namespace std;

int main() {

    ifstream file[9];
    float W_ih1[256*58];
    float b_ih1[256];
    float W_h1h2[256*256];
    float b_h1h2[256];
    float W_h2o[256*23];
    float b_h2o[23];
    float obs_mean[58];
    float obs_var[58];

    float input[58];
    float hidden1[256];
    float hidden2[256];
    float output[23];

    float mocap_data[39][31];

	file[0].open("obs_mean.txt", ios::in);
	file[1].open("obs_variance.txt", ios::in);
	file[2].open("mlp_extractor_policy_net_0_weight.txt", ios::in);
	file[3].open("mlp_extractor_policy_net_0_bias.txt", ios::in);
	file[4].open("mlp_extractor_policy_net_2_weight.txt", ios::in);
	file[5].open("mlp_extractor_policy_net_2_bias.txt", ios::in);
	file[6].open("action_net_weight.txt", ios::in);
	file[7].open("action_net_bias.txt", ios::in);
	file[8].open("/home/kim/red_ws/src/dyros_cc/motions/processed_data_tocabi.txt", ios::in);


	int index = 0;
	float temp;
	if(!file[0].is_open())
	{
		std::cout<<"Can not find the obs_mean file"<<std::endl;
	}
	while(!file[0].eof())
	{
		file[0] >> temp;
		if(temp != '\n')
		{
			obs_mean[index] = temp;
			index ++;
		}
	}

	index = 0;
	if(!file[1].is_open())
	{
		std::cout<<"Can not find the obs_variance file"<<std::endl;
	}
	while(!file[1].eof())
	{
		file[1] >> temp;
		if(temp != '\n')
		{
			obs_var[index] = temp;
			index ++;
		}
	}

	index = 0;
	if(!file[2].is_open())
	{
		std::cout<<"Can not find the mlp_extractor_policy_net_0_weight file"<<std::endl;
	}
	while(!file[2].eof())
	{
		file[2] >> temp;
		if(temp != '\n')
		{
			W_ih1[index] = temp;
			index ++;
		}
	}
  
	index = 0;
	if(!file[3].is_open())
	{
		std::cout<<"Can not find the mlp_extractor_policy_net_0_bias file"<<std::endl;
	}
	while(!file[3].eof())
	{
		file[3] >> temp;
		if(temp != '\n')
		{
			b_ih1[index] = temp;
			index ++;
		}
	}
  
	index = 0;
	if(!file[4].is_open())
	{
		std::cout<<"Can not find the mlp_extractor_policy_net_2_weight file"<<std::endl;
	}
	while(!file[4].eof())
	{
		file[4] >> temp;
		if(temp != '\n')
		{
			W_h1h2[index] = temp;
			index ++;
		}
	}
  
	index = 0;
	if(!file[5].is_open())
	{
		std::cout<<"Can not find the mlp_extractor_policy_net_2_bias file"<<std::endl;
	}
	while(!file[5].eof())
	{
		file[5] >> temp;
		if(temp != '\n')
		{
			b_h1h2[index] = temp;
			index ++;
		}
	}

	index = 0;
	if(!file[6].is_open())
	{
		std::cout<<"Can not find the action_net_weight file"<<std::endl;
	}
	while(!file[6].eof())
	{
		file[6] >> temp;
		if(temp != '\n')
		{
			W_h2o[index] = temp;
			index ++;
		}
	}

	index = 0;
	if(!file[7].is_open())
	{
		std::cout<<"Can not find the action_net_bias file"<<std::endl;
	}
	while(!file[7].eof())
	{
		file[7] >> temp;
		if(temp != '\n')
		{
			b_h2o[index] = temp;
			index ++;
		}
	}

	int row = 0;
	int col = 0;
	if(!file[8].is_open())
	{
		std::cout<<"Can not find the processed_data_tocabi file"<<std::endl;
	}
	while(!file[8].eof())
	{
		file[8] >> temp;
		if(temp != '\n')
		{
			mocap_data[row][col] = temp;
			col ++;
			if (col==31)
			{
				row ++;
				col = 0;
			}
		}
	}

  cout<<"Mocap Data: " << mocap_data[0][0] << " "  << mocap_data[0][30] << " "  << mocap_data[38][0] << " " << mocap_data[38][30] << endl;

  high_resolution_clock::time_point t1 = high_resolution_clock::now();
		// Observation
    float obs[58] = {2.89473676e-01,  1.00000000e+00, -1.02983053e-10, 8.27958292e-11,
   7.00462695e-10,  1.16535049e-09,  2.72295653e-09,  1.82680865e-09,
   2.33153556e-08,  4.84634367e-10, -7.27815759e-10, -1.94878092e-09,
   2.49204591e-09,  2.69588987e-10,  2.65049210e-08, -4.67799657e-10,
   1.18959601e-09, -2.01533970e-10, -3.97165915e-10, -8.87878573e-11,
  -3.61175270e-09,  2.10723694e-09,  1.57000005e+00,  5.02167979e-09,
  -1.34434600e-09,  1.27447241e-09, -1.57000005e+00, -7.23445032e-09,
   4.18528340e-09,  5.83506743e-10,  2.14732142e-09, -9.71044516e-10,
  -2.94650689e-10, -4.02897399e-10,  1.89097247e-10,  2.17658825e-10,
  -4.65857169e-10,  1.55823374e-09,  1.11047338e-09, -9.91262307e-11,
   2.37266518e-10,  2.95553099e-10, -2.50509658e-10,  3.68725312e-10,
   8.24655448e-11, -1.89778982e-10, -3.48705158e-11,  1.83301809e-11,
  -2.40734602e-11, -1.05788380e-10,  3.26518895e-10, -4.14619946e-10,
   3.40306654e-10,  8.95392649e-10,  3.20388787e-11, -3.78118186e-10,
  -9.63738318e-10,  9.70000000e-01,};
		// Normalize
		for(int i=0; i<58; i++)
		{
			input[i] = (obs[i]-obs_mean[i])/sqrt(obs_var[i]+1.0e-08);
		}

		// Network Feedforward
		// Input Layer
		for(int row=0; row<256; row++)
		{
			hidden1[row] = b_ih1[row];
			for(int col=0; col<58; col++)
			{
				hidden1[row] +=  W_ih1[58*row+col] * input[col];
			}
      if (hidden1[row] < 0.0)
        hidden1[row] = 0.0;
		}
		// Hidden Layer
		for(int row=0; row<256; row++)
		{
			hidden2[row] = b_h1h2[row];
			for(int col=0; col<256; col++)
			{
				hidden2[row] +=  W_h1h2[256*row+col] * hidden1[col];
			}
      if (hidden2[row] < 0.0)
        hidden2[row] = 0.0;
		}

		// Output Layer
		for(int row=0; row<23; row++)
		{
			output[row] = b_h2o[row];
			for(int col=0; col<256; col++)
			{
				output[row] +=  W_h2o[256*row+col] * hidden2[col];
			}
		}

high_resolution_clock::time_point t2 = high_resolution_clock::now();
duration<double, std::milli> time_span = t2 - t1;
std::cout << "It took me " << time_span.count() << " milliseconds."<<std::endl;
std::cout << "Output: " << output[0] << " " << output[1]<< std::endl;
}
