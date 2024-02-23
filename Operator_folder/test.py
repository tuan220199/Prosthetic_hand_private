from gforce import GForceProfile
import asyncio



async def main():
    sampRate = 500
    channelMask = 0xFF
    dataLen = 128
    resolution = 8

    while True:
        GF = GForceProfile()

        print("Scanning devices...")

        # Scan all gforces,return [[num,dev_name,dev_addr,dev_Rssi,dev_connectable],...]
        scan_results = await GF.scan(5)

        # Display the first menu
        print('_'*75)
        print('0: exit')

        if scan_results == []:
            print('No bracelet was found')
        else:
            for d in scan_results:
                try:
                    print('{0:<1}: {1:^16} {2:<18} Rssi={3:<3}, connectable:{4:<6}'.format(*d))
                except:
                    pass
            # end for

        # Handle user actions
        button = int(input('Please select the device you want to connect or exit:'))

        if button == 0:
             break
        else:
            addr = scan_results[button-1][2]
            GF.connect(addr)

        #     # Display the secord menu
        #     while True:
        #         time.sleep(1)
        #         print2menu()
        #         button = int(input('Please select a function or exit:'))

        #         if button == 0:
        #             break

        #         elif button == 1:
        #             GF.getControllerFirmwareVersion(get_firmware_version_cb, 1000)

        #         elif button == 2:
        #             GF.setLED(False, set_cmd_cb, 1000)
        #             time.sleep(3)
        #             GF.setLED(True, set_cmd_cb, 1000)

        #         elif button == 3:
        #             GF.setMotor(True, set_cmd_cb, 1000)
        #             time.sleep(3)
        #             GF.setMotor(False, set_cmd_cb, 1000)

        #         elif button == 4:
        #             GF.setDataNotifSwitch(DataNotifFlags['DNF_QUATERNION'], set_cmd_cb, 1000)
        #             time.sleep(1)
        #             GF.startDataNotification(ondata)

        #             button = input()
        #             print("Stopping...")
        #             GF.stopDataNotification()
        #             time.sleep(1)
        #             GF.setDataNotifSwitch(DataNotifFlags['DNF_OFF'], set_cmd_cb, 1000)

        #         elif button == 5:
        #             sampRate = eval(input('Please enter sample value(max 500, e.g., 500): '))
        #             channelMask = eval(input('Please enter channelMask value(e.g., 0xFF): '))
        #             dataLen = eval(input('Please enter dataLen value(e.g., 128): '))
        #             resolution = eval(input('Please enter resolution value(8 or 12, e.g., 8): '))

        #         elif button == 6:
        #             GF.setEmgRawDataConfig(sampRate, channelMask, dataLen, resolution, cb=set_cmd_cb, timeout=1000)
        #             GF.setDataNotifSwitch(DataNotifFlags['DNF_EMG_RAW'], set_cmd_cb, 1000)
        #             time.sleep(1)
        #             GF.startDataNotification(ondata)
                    
        #             button = input()
        #             print("Stopping...")
        #             GF.stopDataNotification()
        #             time.sleep(1)
        #             GF.setDataNotifSwitch(DataNotifFlags['DNF_OFF'], set_cmd_cb, 1000)
        #     # end while

            #break

if __name__ == '__main__':
    asyncio.run(main())