using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.Data;

namespace MachineLearning.Models
{
    public class CasaPredictionResult
    {
        [ColumnName("Score")]
        public float PrecoPrevisto { get; set; }
    }
}