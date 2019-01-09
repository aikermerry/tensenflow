%************************************************************************%
%**
%**   ===========================语音信号预处理============================
%**
%**   --掩蔽处理--DCT变换--静音处理--掩蔽处理--                  
%**   --保留处理后的频域信号--                                  
%**   --生成处理后的音频文件--
%**       
%************************************************************************%

clear;   clc;   close all;
tic

%--$$$$$$$$$$$$$$$$$$$    --- 参数设置---   $$$$$$$$$$$$$$$$$$$ --

a = 1:1:100;    a = a';    set = sprintfc('%g',a);    W = size(set,1);

%----  参数设置：帧长为Rows；  ---%
Rows = 1024;        file1 = 'Audio_Ini2/';  %---音频所在文件夹


%--$$$$$$$$$$$$$$$$$$$    --- 音频读取---   $$$$$$$$$$$$$$$$$$$ --

for w = 1:W    %---音频个数，循环读取音频文件处理
    clear Input_Audio Use_Audio_di Use_Audio Use_Audio_p Tq Location_Zero
    clear tempy dertabb  fin_loc Spl_max Spl_locb Location_Masked 
    
    Input_Audio_num = strcat(set(w),'.WAV');   %字符连接，确定音频文件名
    
    %===============    读取音频文件    =================%
    
    [Input_Audio,FS] = audioread(Input_Audio_num{1});   %---nbit 表示用多少位表示一个语音采样；
    
    %----  提取出一个单声道Input_Audio，并对该声道用TEMP_IN保存  -----%
    Input_Audio = Input_Audio(:,1);     TEMP_IN = Input_Audio;   Naudio = length(Input_Audio);% 信号长度
    
    %===============    写入采样音频    =================%
    
    sample = strcat(file1,'采样',set(w),'.wav');   %字符连接，确定存入的文件下的音频文件名
    
    audiowrite(sample{1},TEMP_IN,FS);
    
    
    %===============  采样音频分帧处理  =================%
    
    MAX_InAudio = max(abs(Input_Audio));	% 找出语音幅度最大值； 
    Columns = ceil(length(Input_Audio)/Rows);  %找出矩阵的列数；
    Input_Audio = [Input_Audio;zeros(Columns*Rows-length(Input_Audio),1)];	 %在矩阵后加0，使其能使用reshape
    Input_Audio = reshape(Input_Audio,Rows,Columns);      %将矩阵转置为需要的列和行

    Input_Audio = Input_Audio./MAX_InAudio;     %---归一化声音幅度；
    
    
    %--$$$$$$$$$$$$$$$$$$$    ---数据块处理，时频分析---   $$$$$$$$$$$$$$$$$$$ --
    Use_Audio = dct(Input_Audio);
    temp_audio = Use_Audio;
    Use_Audio_p  = (10*log10((abs(Use_Audio)).^2));
    Use_Audio_p = Use_Audio_p-max(max(Use_Audio_p))+80;
    
    
    %--$$$$$$$$$$$$$$$$$$$    --- 静音门限处理---   $$$$$$$$$$$$$$$$$$$ --
    
    %==================  静音门限设置   ==================%
    freq = (FS/Rows/1): (FS/Rows/1) : (FS/Rows/1)+ (Rows-1)*(FS/Rows/1);
    bark = hz2bark(freq);                      % array of Bark corresponding to bins  PLOT
    Tq_temp = 3.64*(freq/1000).^(-0.8) - 6.5*exp(-0.6*(freq/1000-3.3).^2) + 10^(-3) * (freq/1000).^4;
    Tq = kron( (Tq_temp).', ones(1,Columns));
    
    %=========== 将低于静音门限的频率幅度置零 =============%
    Location_Zero = find(Use_Audio_p -Tq <=0 );  %----静音门限加权倍数Time_quit = 1;
    Nzero = length(Location_Zero);     % 信号中零的个数
    Use_Audio(Location_Zero) = 0;
    Use_Audio_jinyin1 = Use_Audio;


    K1 = ((Naudio-Nzero)/Naudio)*100   %---稀疏率，Naudio信号长度，Nzero零的个数
    
    %===============   写入稀疏音频信号   =================%
    Use_Audio_di  = idct(Use_Audio);%用于播放通过静音门限处理后的语音
    Use_Audio_di = reshape(Use_Audio_di,Rows*Columns,1);
    sample1 = strcat(file1,'静音',set(w),'.wav');   %字符连接，确定存入的文件下的音频文件名
    audiowrite(sample1{1},Use_Audio_di*MAX_InAudio,FS);
%     sound(Use_Audio_di*MAX_InAudio,FS);
    
    
    %--$$$$$$$$$$$$$$$$$$$    --- 静音门限处理---   $$$$$$$$$$$$$$$$$$$ --
    bark = bark';     bin = 1;     jj = 1;
    
    %====================    临界频带分带     ====================%
    for ii = 1:length(bark)-1
    if bark(ii) <= jj & bark(ii+1)>jj 
%         Use_Audio_P{jj,1} = Use_Audio_p(bin:ii,:);
%         Use_Audio_P{jj,2} = bark(bin:ii,1);
%         bin = ii+1;
        fin_loc(jj,1) = ii;
        jj = jj + 1;
%         if jj == 25
%             Use_Audio_P{jj,1} = Use_Audio_p(bin:length(bark),:);
%             Use_Audio_P{jj,2} = bark(bin:length(bark),1);
%             break
%         end
    end
    end
    
    %==================== 确定带内最大值及其位置 ====================%
    for bb = 1:25
        if bb == 1
            bin = 1;
        else
            bin = fin_loc(bb-1)+1;
        end
        if bb == 25
            wei = Rows;
        else
            wei = fin_loc(bb);
        end
        [Spl_max(bb,:),Spl_locb(bb,:)] = max(Use_Audio_p(bin:wei,:),[],1);
    %     for cc = 1:Columns
    %         [Use_Audio_P{bb,3},Use_Audio_P{bb,4}]= max(Use_Audio_P{bb,1},[],1);
    %     end
    end

    Spl_loc = [Spl_locb(1,:);Spl_locb(2:25,:) + kron( fin_loc, ones(1,Columns))];
    % for cc = 1:Columns
    %     Spl_max1(:,cc) = Use_Audio_p(Spl_loc1(:,cc),cc);
    % end
    
   %====================  确定带内掩蔽阈值  ====================% 
    for cc = 1:Columns
        for bb = 1:25
            if bb == 1
                bin = 1;
            else
                bin = fin_loc(bb-1)+1;
            end
            if bb == 25
                wei = Rows;
            else
                wei = fin_loc(bb);
            end
            dertabb = bark(bin:wei,1)-bark(Spl_loc(bb,cc));
            tempy(bin:wei,cc) = 15.8+7.5*(dertabb+0.474)-17.5*sqrt(1+(dertabb+0.474).^2)+Spl_max(bb,cc)-25;
            L = wei - bin + 1;
    %         tempy(bin:wei,cc) = ones(L,1).*Spl_max(bb,cc)-24; 
        end
    end
    
    %=========== 将低于掩蔽阈值的频率幅度置零 =============%
    Location_Masked = find(Use_Audio_p -tempy <=0 );
    Use_Audio(Location_Masked) = 0;
    Use_Audio_jinyin2 = Use_Audio;

    Nzero = length(find(Use_Audio ==0));
    K2 = ((Naudio-Nzero)/Naudio)*100   %---稀疏率，Naudio信号长度，Nzero零的个数
    
    %================== 写入稀疏音频信号 ==================%
    Use_Audio_di  = idct(Use_Audio);%用于播放通过静音门限处理后的语音
    Use_Audio_di = reshape(Use_Audio_di,Rows*Columns,1);
    sample2 = strcat(file1,'掩蔽',set(w),'.wav');   %字符连接，确定存入的文件下的音频文件名
    audiowrite(sample2{1},Use_Audio_di*MAX_InAudio,FS);
%     sound(Use_Audio_di*MAX_InAudio,FS);
    
    
    %--$$$$$$$$$$$$$$$$$$$    ---PESQ音频评分---   $$$$$$$$$$$$$$$$$$$ --

    %---读入音频文件
    audio1 =  strcat(file1,'采样',set{w},'.wav');
    audio2 =  strcat(file1,'静音',set{w},'.wav');
    audio3 =  strcat(file1,'掩蔽',set{w},'.wav');

    %---评分，评分的音频为所有仿真次数后最终的平均值生成音频进行评价
%     PESQ_MOS1 = composite(audio1, audio2)
%     PESQ_MOS2 = composite(audio1, audio3)
    PESQ_MOS3 = composite(audio2, audio3)
   
    
    %===============    保存最终稀疏化数据    =================%
    %---保存幅值
    variable = strcat('Use_Audio',set{w});      %变量名例如：Use_Audio002
    eval([variable,'=','Use_Audio',';']);  %自动生成变量Use_Audio002，并将Use_Audio赋值给它
%     save('Use_Audio512',variable);    %保存心里模型处理后的信号
    save('Use_Audio512',variable,'-append');    %保存心里模型处理后的信号
    
    K_ATH(w,1) = K1;  %w文件的稀疏率
    K_Mask(w,1) = K2;  %w文件的稀疏率
    MOS(w,1) = PESQ_MOS3;
    
end
save('MosandK1-100','K_ATH','K_Mask','MOS');
toc


