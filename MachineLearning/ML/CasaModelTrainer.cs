using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using MachineLearning.Models;
using Microsoft.ML;

namespace MachineLearning.ML
{
    public class CasaModelTrainer
    {
        private MLContext mLContext = new MLContext();
        private IDataView dados;
        private ITransformer modeloTreinado;

        public void CarregarDadosCSV(string path)
        {
            dados = mLContext.Data.LoadFromTextFile<CasaInputData>(
                path: path,
                hasHeader: true,
                separatorChar: ','
            );
        }

        public void TreinarModelo()
        {
            var pipeline = mLContext.Transforms.Concatenate(
                "Features",
                nameof(CasaInputData.Tamanho),
                nameof(CasaInputData.Quartos)
            ).Append(mLContext.Regression.Trainers.LightGbm(
                labelColumnName: "Preco",
                numberOfIterations: 100
            ));

            modeloTreinado = pipeline.Fit(dados);
        }

        public void SalvarModelo(string path)
        {
            mLContext.Model.Save(modeloTreinado, dados.Schema, path);
        }

        public void AvaliarModelo()
        {
            var previsao = modeloTreinado.Transform(dados);

            var metricas = mLContext.Regression.Evaluate(
                data: previsao,
                labelColumnName: "Preco",
                scoreColumnName: "Score"
            );

            System.Console.WriteLine($"MAE: {metricas.MeanAbsoluteError}");
            System.Console.WriteLine($"RMSE: {metricas.RootMeanSquaredError}");
            System.Console.WriteLine($"R2: {metricas.RSquared}");
        }
    }
}