from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from proxyclip_segmentor import ProxyCLIPSegmentation
import sys

def process_image(input_file, output_file, keywords):
    img = Image.open(input_file)
    name_list = keywords.split(",")

    with open('./configs/my_name.txt', 'w') as writers:
        for i in range(len(name_list)):
            if i == len(name_list) - 1:
                writers.write(name_list[i])
            else:
                writers.write(name_list[i] + '\n')
    writers.close()

    img_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
        #transforms.Resize((400, 400)),
    ])(img)

    img_tensor = img_tensor.unsqueeze(0).to('cuda')


    model = ProxyCLIPSegmentation(clip_type='openai', model_type='ViT-B/16', vfm_model='dino',
                                name_path='./configs/my_name.txt')

    seg_pred = model.predict(img_tensor, data_samples=None)
    seg_pred = seg_pred.data.cpu().numpy().squeeze(0)

    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    ax[0].imshow(img)
    ax[0].axis('off')
    ax[1].imshow(seg_pred, cmap='viridis')
    ax[1].axis('off')
    plt.tight_layout()
    #plt.show()
    plt.savefig(output_file, bbox_inches='tight')

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("用法: python process_image.py <输入文件路径> <输出文件路径> <关键词>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    keywords = sys.argv[3]
    process_image(input_file, output_file, keywords)
