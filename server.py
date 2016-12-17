import binascii
import serial

def serial_ports():
    ports = ['COM%s' % (i+1) for i in range(256)]
    result = []
    for port in ports:
        try:
            serial_check = serial.Serial(port)
            serial_check.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass
    return result
    
COM = serial_ports()
print "Available ports to choose: %s" % COM
SER = serial.Serial('COM4', 115200, timeout=None)
FILE_COUNT = 0
while 1:
    LINE = SER.readline()
    print LINE
    if LINE == "Start\r\n":
        print "Incoming image."
        FILE_COUNT = FILE_COUNT + 1
        FILE_NAME = str(FILE_COUNT)
        F = open('in/'+ FILE_NAME +'.jpg', 'ab')
        FTXT = open('in/' + FILE_NAME + '.txt', 'ab')
        LINE = SER.readline()
        while LINE != "GG\r\n":
            F.write(binascii.unhexlify(LINE.rstrip()))
            #print binascii.unhexlify(LINE.rstrip())
            LINE = SER.readline()
    
    if LINE == "GG\r\n":
        F.close()
        print "Done reading image."
        print "Image written to : " + FILE_NAME + ".jpg"
        print "Image information written to: " + FILE_NAME + ".txt"
        LINE = SER.readline()
    
    if LINE == "Height\r\n":
        LINE = SER.readline()
        FTXT.write((LINE.rstrip()))
        FTXT.close()