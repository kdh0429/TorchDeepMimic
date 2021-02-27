clc
clear all

Deg2Rad = pi/180;
Rad2Deg = 180/pi;

fname = 'humanoid3d_walk.txt';
val = jsondecode(fileread(fname));
data = val.Frames;
total_time = 1.3;

for i=1:size(data,1)
    data_quat2joint(i,1) = (i-1)*3.3332000000000001e-02;
end
% root position and orientation
data_quat2joint(:,2:8) = data(:,2:8);


% left hip rotation
data_quat2joint(:,9:11) = eulerd(quaternion(data(:,31:34)),'YXZ','frame');
data_quat2joint(:,11) = -data_quat2joint(:,11);
% left knee roatation
data_quat2joint(:,12) = -data(:,35)*Rad2Deg;
% left ankle rotation
left_ankle = eulerd(quaternion(data(:,36:39)),'ZXY','frame');
data_quat2joint(:,13) = -left_ankle(:,1);
data_quat2joint(:,14) = left_ankle(:,2);

% right hip rotation
data_quat2joint(:,15:17) = eulerd(quaternion(data(:,17:20)),'YXZ','frame');
data_quat2joint(:,17) = -data_quat2joint(:,17);
% right knee roatation
data_quat2joint(:,18) = -data(:,21)*Rad2Deg;
% right ankle rotation
right_ankle = eulerd(quaternion(data(:,22:25)),'ZXY','frame');
data_quat2joint(:,19) = -right_ankle(:,1);
data_quat2joint(:,20) = right_ankle(:,2);


% chest rotation
data_quat2joint(:,21:23) = eulerd(quaternion(data(:,9:12)),'YZX','frame');
data_quat2joint(:,22) = -data_quat2joint(:,22);


% left shoulder rotation
data_quat2joint(:,24:26) = eulerd(quaternion(data(:,40:43)),'YZX','frame');
data_quat2joint(:,25) = -data_quat2joint(:,25);
data_quat2joint(:,26) = 90;
% left elbow roatation
data_quat2joint(:,27) = -data(:,44)*Rad2Deg;

% right shoulder rotation
data_quat2joint(:,28:30) = eulerd(quaternion(data(:,26:29)),'YZX','frame');
data_quat2joint(:,29) = data_quat2joint(:,29);
data_quat2joint(:,30) = -90;
% right elbow roatation
data_quat2joint(:,31) = data(:,30)*Rad2Deg;


data_quat2joint(:,9:31) = data_quat2joint(:,9:31)*Deg2Rad;


save('processed_data_tocabi.txt', 'data_quat2joint', '-ascii', '-double', '-tabs')
