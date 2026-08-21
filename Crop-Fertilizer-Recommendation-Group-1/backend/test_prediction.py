from services.crop_predictor import predict_crop


result = predict_crop(
    N=90,
    P=42,
    K=43,
    temperature=20.8,
    humidity=82,
    ph=6.5,
    rainfall=202
)

print(result)