%************************************************************************%
%**
%**   ===========================�����ź�Ԥ����============================
%**
%**   --�ڱδ���--DCT�任--��������--�ڱδ���--                  
%**   --����������Ƶ���ź�--                                  
%**   --���ɴ�������Ƶ�ļ�--
%**       
%************************************************************************%

clear;   clc;   close all;
tic

%--$$$$$$$$$$$$$$$$$$$    --- ��������---   $$$$$$$$$$$$$$$$$$$ --

a = 1:1:100;    a = a';    set = sprintfc('%g',a);    W = size(set,1);

%----  �������ã�֡��ΪRows��  ---%
Rows = 1024;        file1 = 'Audio_Ini2/';  %---��Ƶ�����ļ���


%--$$$$$$$$$$$$$$$$$$$    --- ��Ƶ��ȡ---   $$$$$$$$$$$$$$$$$$$ --

for w = 1:W    %---��Ƶ������ѭ����ȡ��Ƶ�ļ�����
    clear Input_Audio Use_Audio_di Use_Audio Use_Audio_p Tq Location_Zero
    clear tempy dertabb  fin_loc Spl_max Spl_locb Location_Masked 
    
    Input_Audio_num = strcat(set(w),'.WAV');   %�ַ����ӣ�ȷ����Ƶ�ļ���
    
    %===============    ��ȡ��Ƶ�ļ�    =================%
    
    [Input_Audio,FS] = audioread(Input_Audio_num{1});   %---nbit ��ʾ�ö���λ��ʾһ������������
    
    %----  ��ȡ��һ��������Input_Audio�����Ը�������TEMP_IN����  -----%
    Input_Audio = Input_Audio(:,1);     TEMP_IN = Input_Audio;   Naudio = length(Input_Audio);% �źų���
    
    %===============    д�������Ƶ    =================%
    
    sample = strcat(file1,'����',set(w),'.wav');   %�ַ����ӣ�ȷ��������ļ��µ���Ƶ�ļ���
    
    audiowrite(sample{1},TEMP_IN,FS);
    
    
    %===============  ������Ƶ��֡����  =================%
    
    MAX_InAudio = max(abs(Input_Audio));	% �ҳ������������ֵ�� 
    Columns = ceil(length(Input_Audio)/Rows);  %�ҳ������������
    Input_Audio = [Input_Audio;zeros(Columns*Rows-length(Input_Audio),1)];	 %�ھ�����0��ʹ����ʹ��reshape
    Input_Audio = reshape(Input_Audio,Rows,Columns);      %������ת��Ϊ��Ҫ���к���

    Input_Audio = Input_Audio./MAX_InAudio;     %---��һ���������ȣ�
    
    
    %--$$$$$$$$$$$$$$$$$$$    ---���ݿ鴦��ʱƵ����---   $$$$$$$$$$$$$$$$$$$ --
    Use_Audio = dct(Input_Audio);
    temp_audio = Use_Audio;
    Use_Audio_p  = (10*log10((abs(Use_Audio)).^2));
    Use_Audio_p = Use_Audio_p-max(max(Use_Audio_p))+80;
    
    
    %--$$$$$$$$$$$$$$$$$$$    --- �������޴���---   $$$$$$$$$$$$$$$$$$$ --
    
    %==================  ������������   ==================%
    freq = (FS/Rows/1): (FS/Rows/1) : (FS/Rows/1)+ (Rows-1)*(FS/Rows/1);
    bark = hz2bark(freq);                      % array of Bark corresponding to bins  PLOT
    Tq_temp = 3.64*(freq/1000).^(-0.8) - 6.5*exp(-0.6*(freq/1000-3.3).^2) + 10^(-3) * (freq/1000).^4;
    Tq = kron( (Tq_temp).', ones(1,Columns));
    
    %=========== �����ھ������޵�Ƶ�ʷ������� =============%
    Location_Zero = find(Use_Audio_p -Tq <=0 );  %----�������޼�Ȩ����Time_quit = 1;
    Nzero = length(Location_Zero);     % �ź�����ĸ���
    Use_Audio(Location_Zero) = 0;
    Use_Audio_jinyin1 = Use_Audio;


    K1 = ((Naudio-Nzero)/Naudio)*100   %---ϡ���ʣ�Naudio�źų��ȣ�Nzero��ĸ���
    
    %===============   д��ϡ����Ƶ�ź�   =================%
    Use_Audio_di  = idct(Use_Audio);%���ڲ���ͨ���������޴���������
    Use_Audio_di = reshape(Use_Audio_di,Rows*Columns,1);
    sample1 = strcat(file1,'����',set(w),'.wav');   %�ַ����ӣ�ȷ��������ļ��µ���Ƶ�ļ���
    audiowrite(sample1{1},Use_Audio_di*MAX_InAudio,FS);
%     sound(Use_Audio_di*MAX_InAudio,FS);
    
    
    %--$$$$$$$$$$$$$$$$$$$    --- �������޴���---   $$$$$$$$$$$$$$$$$$$ --
    bark = bark';     bin = 1;     jj = 1;
    
    %====================    �ٽ�Ƶ���ִ�     ====================%
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
    
    %==================== ȷ���������ֵ����λ�� ====================%
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
    
   %====================  ȷ�������ڱ���ֵ  ====================% 
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
    
    %=========== �������ڱ���ֵ��Ƶ�ʷ������� =============%
    Location_Masked = find(Use_Audio_p -tempy <=0 );
    Use_Audio(Location_Masked) = 0;
    Use_Audio_jinyin2 = Use_Audio;

    Nzero = length(find(Use_Audio ==0));
    K2 = ((Naudio-Nzero)/Naudio)*100   %---ϡ���ʣ�Naudio�źų��ȣ�Nzero��ĸ���
    
    %================== д��ϡ����Ƶ�ź� ==================%
    Use_Audio_di  = idct(Use_Audio);%���ڲ���ͨ���������޴���������
    Use_Audio_di = reshape(Use_Audio_di,Rows*Columns,1);
    sample2 = strcat(file1,'�ڱ�',set(w),'.wav');   %�ַ����ӣ�ȷ��������ļ��µ���Ƶ�ļ���
    audiowrite(sample2{1},Use_Audio_di*MAX_InAudio,FS);
%     sound(Use_Audio_di*MAX_InAudio,FS);
    
    
    %--$$$$$$$$$$$$$$$$$$$    ---PESQ��Ƶ����---   $$$$$$$$$$$$$$$$$$$ --

    %---������Ƶ�ļ�
    audio1 =  strcat(file1,'����',set{w},'.wav');
    audio2 =  strcat(file1,'����',set{w},'.wav');
    audio3 =  strcat(file1,'�ڱ�',set{w},'.wav');

    %---���֣����ֵ���ƵΪ���з�����������յ�ƽ��ֵ������Ƶ��������
%     PESQ_MOS1 = composite(audio1, audio2)
%     PESQ_MOS2 = composite(audio1, audio3)
    PESQ_MOS3 = composite(audio2, audio3)
   
    
    %===============    ��������ϡ�軯����    =================%
    %---�����ֵ
    variable = strcat('Use_Audio',set{w});      %���������磺Use_Audio002
    eval([variable,'=','Use_Audio',';']);  %�Զ����ɱ���Use_Audio002������Use_Audio��ֵ����
%     save('Use_Audio512',variable);    %��������ģ�ʹ������ź�
    save('Use_Audio512',variable,'-append');    %��������ģ�ʹ������ź�
    
    K_ATH(w,1) = K1;  %w�ļ���ϡ����
    K_Mask(w,1) = K2;  %w�ļ���ϡ����
    MOS(w,1) = PESQ_MOS3;
    
end
save('MosandK1-100','K_ATH','K_Mask','MOS');
toc


