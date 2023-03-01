﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;

namespace BrightData.Cuda.CudaToolkit
{
    /// <summary>
    /// An CudaBlasException is thrown, if any wrapped call to the CUBLAS-library does not return <see cref="CublasStatus.Success"/>.
    /// </summary>
    public class CudaBlasException : Exception, ISerializable
    {

        private CublasStatus _cudaBlasError;

        #region Constructors
        /// <summary>
        /// 
        /// </summary>
        public CudaBlasException()
        {

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="serInfo"></param>
        /// <param name="streamingContext"></param>
        protected CudaBlasException(SerializationInfo serInfo, StreamingContext streamingContext)
            : base(serInfo, streamingContext)
        {
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="error"></param>
        public CudaBlasException(CublasStatus error)
            : base(GetErrorMessageFromCUResult(error))
        {
            this._cudaBlasError = error;
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="message"></param>
        public CudaBlasException(string message)
            : base(message)
        {

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="message"></param>
        /// <param name="exception"></param>
        public CudaBlasException(string message, Exception exception)
            : base(message, exception)
        {

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="error"></param>
        /// <param name="message"></param>
        /// <param name="exception"></param>
        public CudaBlasException(CublasStatus error, string message, Exception exception)
            : base(message, exception)
        {
            this._cudaBlasError = error;
        }
        #endregion

        #region Methods
        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            return this._cudaBlasError.ToString();
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="info"></param>
        /// <param name="context"></param>
        public override void GetObjectData(SerializationInfo info, StreamingContext context)
        {
            base.GetObjectData(info, context);
            info.AddValue("CudaBlasError", this._cudaBlasError);
        }
        #endregion

        #region Static methods
        private static string GetErrorMessageFromCUResult(CublasStatus error)
        {
            string message = string.Empty;

            switch (error)
            {
                case CublasStatus.Success:
                    message = "Any CUBLAS operation is successful.";
                    break;
                case CublasStatus.NotInitialized:
                    message = "The CUBLAS library was not initialized.";
                    break;
                case CublasStatus.AllocFailed:
                    message = "Resource allocation failed.";
                    break;
                case CublasStatus.InvalidValue:
                    message = "An invalid numerical value was used as an argument.";
                    break;
                case CublasStatus.ArchMismatch:
                    message = "An absent device architectural feature is required.";
                    break;
                case CublasStatus.MappingError:
                    message = "An access to GPU memory space failed.";
                    break;
                case CublasStatus.ExecutionFailed:
                    message = "An access to GPU memory space failed.";
                    break;
                case CublasStatus.InternalError:
                    message = "An internal operation failed.";
                    break;
                case CublasStatus.NotSupported:
                    message = "Error: Not supported.";
                    break;
                default:
                    break;
            }


            return error.ToString() + ": " + message;
        }
        #endregion

        #region Properties
        /// <summary>
        /// 
        /// </summary>
        public CublasStatus CudaBlasError
        {
            get
            {
                return this._cudaBlasError;
            }
            set
            {
                this._cudaBlasError = value;
            }
        }
        #endregion
    }
}
