import numpy as np
from qiskit import *
from qiskit import QuantumCircuit,transpile
from PIL import Image,ImageDraw,ImageFont
import math
from math import pi
def quantum(inp_path,out_path,block_size,font_size,magnitude,vals="!@#$%&*()_+",use_color=False):
    try:
        input_image=Image.open(inp_path)
    except FileNotFoundError:
        print("The requested image is not found!")
        return 
    
    input_width,input_height=input_image.size
    rows=input_height//block_size
    cols=input_width//block_size
    #output dim
    char_height=font_size*0.5
    char_width=font_size
    output_width=math.ceil(cols*char_width)
    output_height=math.ceil(rows*char_height)
    #initializing a new canvas
    bgcolor="black"
    output=Image.new("RGB",(output_width,output_height),bgcolor)
    draw=ImageDraw.Draw(output)
    simulator=Aer.get_backend('qasm_simulator')
    def get_brightness(classical_brightness):

        num_qubits=2
        qc=QuantumCircuit(num_qubits,num_qubits)
        theta=(classical_brightness/255.0)*(pi/2)
        qc.ry(theta,0)
        qc.ry(theta,1)
        if(magnitude)>0:
            qc.h(0)
            qc.cry((classical_brightness/255.0)*(pi/4),0,1)
        qc.measure(range(num_qubits),range(num_qubits))
        final=transpile(qc,simulator)
        job=simulator.run(final,shots=10)
        result=job.result()
        counts=result.get_counts(qc)
        
        #now lets convert the count values back to brightness
        if counts:
            res=list(counts.keys())[0]
            #convert from binary to int and then normalize to adjust
            normalization_factor=(int(res,2)/(2**num_qubits-1)/2)
            adjustment_factor=(normalization_factor*(40/2**num_qubits-1))
            adjusted_brightness=min(max(classical_brightness+adjustment_factor,0),255)

            #USE LERP TO BLEND: general formula: a * (1 - t) + b * t(by a factor of t) in our case t=magnitude
            updated_brightness=int(classical_brightness*(1-magnitude)+adjusted_brightness*(magnitude))
            return updated_brightness

#input image processing
#use the get_brightness for each r,g,b values

    '''perceived brightness formulas:
    1)0.2126*R + 0.7152*G + 0.0722*B
    2)0.299*R + 0.587*G + 0.114*B'''
    #Lets use 1
    '''also the input image of various dimensions can lead to more compute time 
    hence we can resize them to a specific size "without ruining it"   ''' 
    input_image2=input_image.resize((cols,rows),Image.Resampling.LANCZOS)

    '''perceived brightness formulas:
        1)0.2126*R + 0.7152*G + 0.0722*B
        2)0.299*R + 0.587*G + 0.114*B
        using 1 '''
 
    for x in range(rows):
        for y in range (cols):
            if use_color:
                r,g,b=input_image2.convert("RGB").getpixel((x,y))
                brightness=(0.2126*r+0.7152*g+0.0722*b)
            else:
                brightness=input_image2.convert("L").getpixel((x,y))
            
            output_brightness=get_brightness(brightness)

            #converting brightness into charecters values based on intensity 
            char_id=min(int(output_brightness/255*(len(ascii_set)-1)),len(ascii_set)-1)
            ascii_val=ascii_set[char_id] 
            pos_x=int(y*char_width)
            pos_y=int(x*char_height)

            if input_image.mode=="RGB":
                r_new=get_brightness(r)
                g_new=get_brightness(g)
                b_new=get_brightness(b)
                fill_color=(r_new,g_new,b_new)
            else:
                if bgcolor=="black":
                    fill_color=(output_brightness,output_brightness,output_brightness)
                else:
                    fill_color=(255-output_brightness,255-output_brightness,255-output_brightness)
            
            #drawing with ascii chars now
            draw.text((pos_x,pos_y),ascii_val,fill=fill_color)
    output.save(out_path) 
    print("Your desired image is saved!!")
    return output

if __name__=="__main__":
    block_size=6
    magnitude=0.22
    font_size=10
    use_color=True
    inp_path='test.png'
    output_path='result.png'
    ascii_set=" .'`^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"     
    quantum(inp_path,output_path,block_size,font_size,magnitude,use_color=use_color)
