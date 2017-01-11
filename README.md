# ocr_recognition   
   验证码识别,该模型是基于xlvector模型上进行加工,验证码内容包含了大小字母以及数字,采用lstm+warp-ctc+cnn达到不分割字符而识别验证码内容～  
   验证码识别,该模型是基于xlvector编写的一段识别数字的代码上进行加工,验证码内容包含了大小字母以及数字,采用lstm+warp-ctc+cnn达到不分割字符而识别验证码>内容～ 
  几点说明：  
 1. 该模型是基于mxnet框架训练而来，基于环境为ubuntu 14，支持GPU和CPU两种模式，如果要运行该代码，需要具备如下软件支持：   
       1. opencv  
       2. openblas   
       3. torch  
       4. cmake  
       5. mxnet  
       6. warp-ctc   
       7. python2.7   
       8. gcc(如果版本太低，要么去掉warp-ctc对应mk目录下的std11标识符，改为std0即可)   
 2. 对于代码的相应的描述：   
       ocr_train.py     为训练模型文件,可以微调模型  
       ocr_predict.py   训练好的模型进行训练   
       lstm_model.py    分装的mx.model，值实现了前馈网络.   
       generator.py     该代码自动生成验证码(为了节约时间，直接摘自网络，再次鸣谢作者).   
       lstm.py          ctc算法处理数据      
 3.验证码效果：  
  ![image](https://github.com/gongxijun/ocr_recognition/prob/master/img_data/iamge/0_1SbM.jpg)  
 4.实际效果   
 Predicted number: C888  实际值： C888  
 Predicted number: CKCX  实际值： GKGX  
 Predicted number: dEpw  实际值： dEpw  
 Predicted number: 2586  实际值： 2586  
 Predicted number: GEvZ  实际值： CEvZ  
 Predicted number: GMXz  实际值： OwKz  
 Predicted number: YLSc  实际值： VjSc  
 Predicted number: WwhG  实际值： Wwh0  
 Predicted number: U4p  实际值： U2AJ   
 Predicted number: vz6C  实际值： yz6G  
 Predicted number: F5l0  实际值： FRl0  
 Predicted number: 3039  实际值： 3039   
 Predicted number: 6756  实际值： 6756  
 Predicted number: BsX5  实际值： DCX5  
 Predicted number: 5m3y  实际值： 5m3y  
 Predicted number: 0BRd  实际值： OBRd  
 Predicted number: 9133  实际值： 9133  
