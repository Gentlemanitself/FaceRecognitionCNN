%%针对图片名为连续标号的Matlab程序
img_list = dir(fullfile('J:\new_data\','*.jpg'));%获取该文件夹中所有jpg格式的信息
fea = [];%特征
gnd = [];%所属类别
num_perClass=15;%每一类包含的图片张数
for i = 1:size(img_list,1)
      img=imread(strcat('J:\new_data\',img_list(i).name));%此处‘.jpg’根据需求更改
      im = reshape(img,1,size(img,1)*size(img,2));
      fea(i,:) = im;
      class = fix(i/num_perClass)+1;%不能整除时，类别数比真实类别差1
      if(class)
          gnd(i,1) = class;
          if(~rem(i,num_perClass)) %如果余数为0，则类别直接是商
             gnd(i,1) = i/num_perClass;
          end
      end
end
save test.mat fea gnd