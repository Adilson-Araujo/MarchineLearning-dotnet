using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using MachineLearning.Models;
using Microsoft.ML;

namespace MachineLearning.ML
{
    public class CasaModelPredictor
    {
        private MLContext mLContext = new MLContext();
        private ITransformer modeloCarregado;

        public void CarregarModelo(string path)
        {
            DataViewSchema modeloSchema;
            modeloCarregado = mLContext.Model.Load(path, out modeloSchema);
        }

        public CasaPredictionResult Prever(CasaInputData novaCasa)
        {
            var predEngine = mLContext.Model.CreatePredictionEngine<CasaInputData, CasaPredictionResult>(
                modeloCarregado
            );

            return predEngine.Predict(novaCasa);
        }
    }
}