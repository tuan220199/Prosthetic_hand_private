packet_cnt = 0
start_time = 0

def ondata(data):
    global ACTION
    global channels
    if len(data) > 0:
        #print('[{0}] data.length = {1}, type = {2}'.format(time.time(), len(data), data[0]))

        if data[0] == NotifDataType['NTF_QUAT_FLOAT_DATA'] and len(data) == 17:
            quat_iter = struct.iter_unpack('f', data[1:])
            quaternion = []
            for i in quat_iter:
                quaternion.append(i[0])
            #end for
            print('quaternion:', quaternion)

        elif data[0] == NotifDataType['NTF_EMG_ADC_DATA'] and len(data) == 129:
            # Data for EMG CH0~CHn repeatly.
            # Resolution set in setEmgRawDataConfig:
            #   8: one byte for one channel
            #   12: two bytes in LSB for one channel.
            # eg. 8bpp mode, data[1] = channel[0], data[2] = channel[1], ... data[8] = channel[7]
            #                data[9] = channel[0] and so on
            # eg. 12bpp mode, {data[2], data[1]} = channel[0], {data[4], data[3]} = channel[1] and so on
 
            # # end for
            extracted_data = data[1:]
            channels += extracted_data
            file1.write(', '.join(map(str, extracted_data)) +', ' + str(ACTION) +"\n")
            #print('\n ------- \n')
            global packet_cnt
            global start_time

            if start_time == 0:
                start_time = time.time()
            
            packet_cnt += 1
            
            if time.time() - start_time > 5:
                ACTION += 1
                print(', '.join(map(str, extracted_data)))
                print('perform action', ACTION, '\n')
                
     
                start_time = time.time()