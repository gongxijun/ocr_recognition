# ocr_recognition
验证码识别,该模型是基于xlvector编写的一段识别数字的代码上进行加工,验证码内容包含了大小字母以及数字,采用lstm+warp-ctc+cnn达到不分割字符而识别验证码内容，对与粘连的验证码以及变形的字符均有较好的效果～  
#几点说明：  
  1. 该模型是基于mxnet框架训练而来，基于环境为ubuntu 14，支持GPU和CPU两种模式，如果要运行该代码，需要具备如下软件支持：   
     1. opencv   
     2. openblas   
     3. torch   
     4. cmake   
     5. mxnet   
     6. warp-ctc   
     7. python2.7   
     8. gcc(如果版本太低，要么去掉warp-ctc对应mk目录下的std11标识符，改为std0即可)   
  2. 对于代码的相应的描述：    
     ocr_train.py     为训练模型文件,可以微调模型   
     ocr_predict.py   训练好的模型进行训练   
     lstm_model.py    分装的mx.model，值实现了前馈网络.   
     generator.py     该代码自动生成验证码(为了节约时间，直接摘自网络，再次鸣谢作者).    
     lstm.py          ctc算法处理数据     
            
  3.效果    
  验证码效果  
    在这次训练的准确率达到56%的时候，进行识别的效果：   
    
 4.在实际中识别中的表现：  
158  
Predicted number: 2479  实际值： 2479   
159   
Predicted number: 2897  实际值： 2897   
160     
Predicted number: 4677  实际值： 4677    
161    
Predicted number: 9153  实际值： 9153    
161    
Predicted number: TZUX  实际值： l7UX    
161    
Predicted number: uUu7  实际值： ulu7     
161    
Predicted number: P3Gh  实际值： PV0h    
162     
Predicted number: 3439  实际值： 3439    
162    
Predicted number: s639  实际值： 5439    
162    
Predicted number: 70Ut  实际值： 7oUt   
163    
Predicted number: 1819  实际值： 1819    
163    
Predicted number: 3665  实际值： 3065    
163    
Predicted number: BCaW  实际值： BGaW    
164    
Predicted number: 3686  实际值： 3686     
164    
Predicted number: b8M7  实际值： H8H7    
165     
Predicted number: 0666  实际值： 0666    
166    
Predicted number: cHJs  实际值： cHJs     
167     
Predicted number: 7390  实际值： 7390     
168    
Predicted number: 8956  实际值： 8956    
168    
Predicted number: 928m  实际值： 9kfm     
169     
Predicted number: 4165  实际值： 4165    
169     
Predicted number: 920  实际值： 3920    
169     
Predicted number: Gb0C  实际值： GM0G    
170    
Predicted number: 1816  实际值： 1816    
170     
Predicted number: i1Dq  实际值： dlBq     
170     
Predicted number: tWoZ  实际值： tVoZ     
170    
Predicted number: vASn  实际值： wWSn     
170     
Predicted number: Jcm  实际值： PTcn    
171    
Predicted number: 1526  实际值： 1526     
171    
Predicted number: caiJ  实际值： acjL    
172    
Predicted number: 5421  实际值： 5421    
172    
Predicted number: VhKe  实际值： VHKe    
173     
Predicted number: 8573  实际值： 8573    
173    
Predicted number: 0268  实际值： 0266     
174     
Predicted number: b0yL  实际值： b0yL     
175    
Predicted number: 0y0w  实际值： 0y0w    
176    
Predicted number: 4259  实际值： 4259     
177     
Predicted number: PTgd  实际值： PTgd     

