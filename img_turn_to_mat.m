%%���ͼƬ��Ϊ������ŵ�Matlab����
img_list = dir(fullfile('J:\new_data\','*.jpg'));%��ȡ���ļ���������jpg��ʽ����Ϣ
fea = [];%����
gnd = [];%�������
num_perClass=15;%ÿһ�������ͼƬ����
for i = 1:size(img_list,1)
      img=imread(strcat('J:\new_data\',img_list(i).name));%�˴���.jpg�������������
      im = reshape(img,1,size(img,1)*size(img,2));
      fea(i,:) = im;
      class = fix(i/num_perClass)+1;%��������ʱ�����������ʵ����1
      if(class)
          gnd(i,1) = class;
          if(~rem(i,num_perClass)) %�������Ϊ0�������ֱ������
             gnd(i,1) = i/num_perClass;
          end
      end
end
save test.mat fea gnd